import paramiko
import time
import os
import uuid
import threading
from typing import Union

class SSHClient:
    def __init__(self, hostname, username, key_file_path, working_directory=None):
        """Initialize the SSH client with the given credentials and optional working directory."""
        self.hostname = hostname
        self.username = username
        self.key_file_path = key_file_path
        self.working_directory = working_directory
        self.ssh_client = None
        self.configure_command = []

    def connect(self):
        """Establish SSH connection."""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Use the private key for authentication if provided
        if self.key_file_path:
            private_key = paramiko.RSAKey.from_private_key_file(self.key_file_path)
            self.ssh_client.connect(hostname=self.hostname, username=self.username, pkey=private_key)
        else:
            self.ssh_client.connect(hostname=self.hostname, username=self.username)

        # Change to the specified working directory, if provided
        if self.working_directory:
            self.change_directory(self.working_directory)

    def change_directory(self, directory):
        """Set the working directory for future commands."""
        self.working_directory = directory

    def add_configure_command(self, command:Union[str,list]):
        """Execute command to configure the shell before running other commands."""
        self.configure_command.append(command)

    def execute_command(self, command, timeout=10):
        """
        Execute a shell command on the remote host.

        Args:
            command (str): The shell command to execute.
            timeout (int): Maximum time in seconds to wait for the command to complete.

        Returns:
            str: The standard output from the command execution.

        Raises:
            Exception: If the command execution fails.
        """
        if self.ssh_client is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        # Prepare the command to source the profile and change directory
        full_command = f"source /etc/profile; source ~/.bashrc; "
        if self.working_directory:
            full_command += f"cd {self.working_directory}; "
            full_command += ' '.join(self.configure_command) + ';'
        full_command += command

        stdin, stdout, stderr = self.ssh_client.exec_command(full_command, timeout=timeout)

        # Wait for the command to complete
        exit_status = stdout.channel.recv_exit_status()

        output = stdout.read().decode('utf-8').strip()
        error_output = stderr.read().decode('utf-8').strip()

        if exit_status == 0:
            return output
        else:
            raise Exception(f"Command execution failed with exit status {exit_status}:\n{error_output}")

    def transfer_file_to_host(self, local_path=None, remote_path=None, content=None):
        """
        Transfer a file or content from the local machine to the remote host.

        Args:
            local_path (str): The path to the local file to transfer.
            remote_path (str): The path on the remote host where the file will be saved.
            content (str): The content to write to the remote file.

        Raises:
            Exception: If the SSH connection is not open or an error occurs during transfer.
        """
        if self.ssh_client is None:
            raise Exception("SSH connection is not open. Call connect() first.")
        if remote_path is None:
            raise ValueError("Remote path must be specified.")

        sftp = self.ssh_client.open_sftp()
        try:
            if content is not None:
                # Write the content directly to the remote file
                with sftp.file(remote_path, 'w') as remote_file:
                    remote_file.write(content)
            elif local_path is not None:
                # Transfer the file from local_path to remote_path
                sftp.put(local_path, remote_path)
            else:
                raise ValueError("Either local_path or content must be provided.")

            print(f"Content successfully transferred to '{remote_path}' on the remote host.")
        except Exception as e:
            raise Exception(f"Error transferring content to host: {e}")
        finally:
            sftp.close()

    def transfer_file_from_host(self, remote_path, local_path):
        """Transfer a file from the remote host to the local machine."""
        if self.ssh_client is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
            print(f"File '{remote_path}' successfully transferred to '{local_path}' on the local machine.")
        except Exception as e:
            raise Exception(f"Error transferring file from host: {e}")
        finally:
            sftp.close()

    def sync_folders(self, remote_folder, local_folder):
        """
        Synchronize a remote folder to a local folder by transferring all files.
        If duplicates exist, files will be overwritten.

        Args:
            remote_folder (str): The path of the remote folder to sync from.
            local_folder (str): The path of the local folder to sync to.
        """
        import os

        if self.ssh_client is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        sftp = self.ssh_client.open_sftp()

        try:
            # Ensure the local folder exists
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)

            # List files in the remote folder
            remote_files = sftp.listdir(remote_folder)

            # Transfer each file to the local folder
            for remote_file in remote_files:
                remote_path = os.path.join(remote_folder, remote_file)
                local_path = os.path.join(local_folder, remote_file)

                try:
                    # Transfer the file from remote to local
                    sftp.get(remote_path, local_path)
                    print(f"File '{remote_file}' synced to local folder.")
                except Exception as e:
                    print(f"Error syncing file '{remote_file}': {e}")

        except Exception as e:
            raise Exception(f"Failed to synchronize folders: {e}")
        finally:
            sftp.close()

    def cleanup(self, files_to_delete, remote_directory=None):
        """
        Delete specified files in the given directory on the remote host.

        Args:
            files_to_delete (list): List of filenames to delete.
            remote_directory (str): Directory where the files are located. Defaults to the working directory.
        """
        if self.ssh_client is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        # Build the command to delete the files
        file_list = ' '.join(files_to_delete)
        command = f"rm -f {file_list}"

        # If a remote directory is specified, change to that directory first
        if remote_directory:
            command = f"cd {remote_directory} && {command}"
        elif self.working_directory:
            command = f"cd {self.working_directory} && {command}"

        # Execute the command on the remote host
        self.execute_command(command)
        print(f"Successfully deleted files: {', '.join(files_to_delete)}")

    def close(self):
        """Close SSH connection."""
        if self.ssh_client is not None:
            self.ssh_client.close()

class PythonSSHClient(SSHClient):
    def __init__(self, hostname, username, key_file_path, working_directory=None, env=None):
        super().__init__(hostname, username, key_file_path, working_directory)
        self.env = env  # The conda environment to activate
        self.python_session = None
        self.output_buffer = ''
        self.output_lock = threading.Lock()
        self.session_active = False

    def connect(self):
        """Establish SSH connection and start Python interactive session."""
        super().connect()

        # Start an interactive shell session
        self.python_session = self.ssh_client.invoke_shell()
        self.session_active = True

        # Wait for the shell to initialize
        time.sleep(1)
        self.clear_buffer()

        # Change to the specified working directory, if provided
        if self.working_directory:
            self.change_directory_python(self.working_directory)

        # Activate the Conda environment, if specified
        if self.env:
            self.activate_env(self.env)

        # Start the Python interpreter
        self.python_session.send('python\n')
        time.sleep(1)
        self.clear_buffer()

        # Start a thread to read output from the session
        self.output_thread = threading.Thread(target=self.read_output)
        self.output_thread.daemon = True
        self.output_thread.start()

    def activate_env(self, env):
        """Activate the specified Conda environment."""
        # Adjust the path to conda.sh according to your Conda installation
        conda_init = "~/miniconda3/etc/profile.d/conda.sh"  # Update this path as needed
        command = f"source {conda_init} && conda activate {env}"
        self.python_session.send(command + '\n')
        time.sleep(1)
        self.clear_buffer()

    def change_directory_python(self, directory):
        """Change the working directory."""
        command = f"cd {directory}"
        self.python_session.send(command + '\n')
        time.sleep(1)
        self.clear_buffer()

    def clear_buffer(self):
        """Clear any pending output from the buffer."""
        if self.python_session.recv_ready():
            self.python_session.recv(1024)

    def read_output(self):
        """Continuously read output from the Python session."""
        while self.session_active:
            if self.python_session.recv_ready():
                data = self.python_session.recv(1024).decode('utf-8')
                with self.output_lock:
                    self.output_buffer += data
            else:
                time.sleep(0.1)

    def invoke(self, code, timeout=10):
        """
        Send code to be executed in the persistent Python session and return the output.

        Args:
            code (str): The Python code to execute.
            timeout (int): Maximum time in seconds to wait for the command to complete.

        Returns:
            str: The output from the execution.

        Raises:
            Exception: If the code execution fails or times out.
        """
        if self.python_session is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        # Generate a unique marker to identify the end of the output
        end_marker = str(uuid.uuid4())
        code_with_marker = f"{code}\nprint(\"{end_marker}\")\n"

        # Clear the output buffer
        with self.output_lock:
            self.output_buffer = ''

        # Send the code to the Python interpreter
        self.python_session.send(code_with_marker + '\n')

        # Wait for the end marker to appear in the output
        start_time = time.time()
        output = ''
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Command execution timed out.")

            with self.output_lock:
                if end_marker in self.output_buffer:
                    output = self.output_buffer
                    # Clear the buffer up to the end marker
                    self.output_buffer = self.output_buffer.split(end_marker, 1)[1]
                    break

            time.sleep(0.1)

        # Process the output to remove prompts and the end marker
        output = output.replace(end_marker, '').strip()

        # Check for errors
        if "Traceback (most recent call last):" in output:
            raise Exception(f"Error during execution:\n{output}")

        return output

    def execute_file(self, remote_file_path, timeout=10):
        """
        Execute a Python file in the persistent Python session.

        Args:
            remote_file_path (str): The path to the Python file on the remote host.
            timeout (int): Maximum time in seconds to wait for the command to complete.

        Returns:
            str: The output from the execution.

        Raises:
            Exception: If the execution fails or times out.
        """
        if self.python_session is None:
            raise Exception("SSH connection is not open. Call connect() first.")

        # Generate a unique marker to identify the end of the output
        end_marker = str(uuid.uuid4())

        # Command to execute the file
        command = f"exec(open('{remote_file_path}').read())\nprint('{end_marker}')"

        # Clear the output buffer
        with self.output_lock:
            self.output_buffer = ''

        # Send the command to the Python interpreter
        self.python_session.send(command + '\n')

        # Wait for the end marker to appear in the output
        start_time = time.time()
        output = ''
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Command execution timed out.")

            with self.output_lock:
                if end_marker in self.output_buffer:
                    output = self.output_buffer
                    # Clear the buffer up to the end marker
                    self.output_buffer = self.output_buffer.split(end_marker, 1)[1]
                    break

            time.sleep(0.1)

        # Process the output to remove prompts and the end marker
        output = output.replace(end_marker, '').strip()

        # Check for errors
        if "Traceback (most recent call last):" in output:
            raise Exception(f"Error during execution:\n{output}")

        return output

    def close(self):
        """Close SSH connection and the Python session."""
        self.session_active = False
        if self.python_session is not None:
            self.python_session.send("exit()\n")  # Exit Python interpreter
            time.sleep(1)
            self.python_session.close()
        super().close()



if __name__ == "__main__":

    python_ssh_client = PythonSSHClient(
        hostname="mariana.matter.sandbox",
        username="yunhengzou",
        key_file_path=None,  # Path to your private key file
        working_directory="/u/yunhengzou/el-agente/",
        env="el-agente" )

    python_ssh_client.connect()

    # module load orca
    python_ssh_client.add_configure_command("module load orca/6.0.1")
    import time
    # count time
    start = time.time()
    print(python_ssh_client.execute_command("/w/185/opt/orca/orca601/orca caffein.inp > caffeine.out"))
    print("Time taken: ", time.time() - start)
    python_ssh_client.close()



    # import tempfile
    # # Initialize SSH client
    ssh_client = SSHClient(
        hostname="mariana.matter.sandbox",  # Replace with your remote host
        username="yunhengzou",             # Replace with your SSH username
        key_file_path=None,                 # Path to your private key file, if needed
        working_directory="/u/yunhengzou/el-agente/"  # Remote working directory
    )

    # Connect to the remote machine
    try:
        ssh_client.connect()
        print("SSH connection established successfully.")
    except Exception as e:
        print(f"Failed to connect via SSH: {e}")

    # Execute a shell command on the remote machine
    try:
        command_output = ssh_client.execute_command("echo 'Hello SSH'")
        print("Shell Command Output:")
        print(command_output)
    except Exception as e:
        print(f"Error executing shell command: {e}")

    # Transfer a local file to the remote machine
    try:
        # Create a temporary local file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_local:
            tmp_local.write("This is a test file for SSH transfer.")
            local_file_path = tmp_local.name

        remote_file_path = "/u/yunhengzou/el-agente/test_transfer.txt"
        ssh_client.transfer_file_to_host(local_path=local_file_path, remote_path=remote_file_path)
        print(f"File '{local_file_path}' successfully transferred to '{remote_file_path}' on the remote host.")
    except Exception as e:
        print(f"Error transferring file to host: {e}")
    finally:
        # Clean up local temporary file
        if 'local_file_path' in locals():
            os.remove(local_file_path)

    # Synchronize a remote folder to a local folder
    try:
        # Create a temporary local directory
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            remote_folder = "/u/yunhengzou/el-agente/"
            local_folder = tmp_local_dir

            ssh_client.sync_folders(remote_folder=remote_folder, local_folder=local_folder)
            print(f"Remote folder '{remote_folder}' synchronized to local folder '{local_folder}'.")

            # List synced files
            synced_files = os.listdir(local_folder)
            print("Synced Files:")
            for file in synced_files:
                print(f" - {file}")
    except Exception as e:
        print(f"Error synchronizing folders: {e}")

    # Perform cleanup by deleting specific files on the remote host
    try:
        files_to_delete = [
            "test_transfer.txt",  # The file we transferred
            "nonexistent_file.txt"  # A file that doesn't exist
        ]
        ssh_client.cleanup(files_to_delete)
        print(f"Cleanup: Successfully deleted files: {', '.join(files_to_delete)}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    # Close the SSH connection
    try:
        ssh_client.close()
        print("SSH connection closed successfully.")
    except Exception as e:
        print(f"Error closing SSH connection: {e}")


    """Python SSH test"""
    # # Initialize Python SSH client
    python_ssh_client = PythonSSHClient(
        hostname="mariana.matter.sandbox",
        username="yunhengzou",
        key_file_path=None,  # Path to your private key file
        working_directory="/u/yunhengzou/el-agente/",
        env="el-agente" 
    )
    # # Connect to the remote machine
    # python_ssh_client.connect()

    # # Execute Python code on the remote machine
    # try:
    #     result = python_ssh_client.invoke("print('hello remote')")
    #     print("Python Code Output:")
    #     print(result)
    # except Exception as e:
    #     print(f"Error executing Python code: {e}")

    # # Close the SSH connection
    # python_ssh_client.close()
