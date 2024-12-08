import os
from langchain.tools import tool
import subprocess
from xyz2graph import MolGraph
from typing import List,Union

@tool
def rdkit_xyz_generation(smiles: str) -> str:
    """Generates XYZ files from SMILES strings using RDKit for 3D structure generation."""
    # generate a dummy xyz file
    print("Generating XYZ file using RDKit...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdmolfiles import MolToXYZFile

    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Optimize geometry
    AllChem.UFFOptimizeMolecule(mol)

    output_xyz = "rdkit_generated.xyz"
    MolToXYZFile(mol, output_xyz)
    #TOOD: run on ssh client 
    print("XYZ file has been generated and stored as rdkit_generated.xyz")
    return f"xyz_file has been generated and stored as rdkit_generated.xyz"

@tool
def xtb_geometry_optimization(input_xyz_file_path: str) -> str:
    """Optimizes molecular geometry using xTB for fast, approximate quantum calculations."""
    # Validate input file format
    print("Running xTB geometry optimization...")
    if not input_xyz_file_path.endswith(".xyz"):
        raise  Exception("Error: Please provide a valid XYZ file for geometry optimization.")
    
    # Check if the file exists
    if not os.path.isfile(input_xyz_file_path):
        raise  Exception(f"Error: The file {input_xyz_file_path} does not exist.")
    
    try:
        # Run xTB geometry optimization
        result = subprocess.run(
            ["xtb", input_xyz_file_path, "--opt"],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        # Check if xTB execution was successful
        if result.returncode != 0:
            raise  Exception(f"Error: xTB failed with the following error:\n{result.stderr}")
        
        # Rename the optimized file if it exists
        optimized_file = "xtbopt.xyz"
        renamed_file = "xtb_optimized.xyz"
        if os.path.exists(optimized_file):
            os.rename(optimized_file, renamed_file)
            print(f"Success: Geometry optimization completed. Optimized file saved as {renamed_file}.")
            return f"Success: Geometry optimization completed. Optimized file saved as {renamed_file}."
        else:
            raise  Exception("Error: xTB did not produce an optimized file.")
    
    except FileNotFoundError:
        return "Error: xTB is not installed or not found in PATH."
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

@tool
def functional_selector(molecule: str) -> str:
    """Selects the appropriate functional for DFT calculations using an ML-based recommendation model."""
    # I think I will use perplexity to search for suitable ones as for now 
    return "B3LYP functional is recommended for the given molecule."

@tool
def basis_set_selector(molecule: str) -> str:
    """Determines the most suitable basis set for DFT calculations based on the molecular system."""
    # I think I will use perplexity to search for suitable ones as for now 
    return "6-311+G(d,p) basis set is recommended for the given molecule."

@tool
def input_file_generator(molecule: str, functional: str, basis_set: str) -> str:
    """Prepares ORCA input files for DFT calculations, integrating selected functionals and basis sets."""
    with open(f"{molecule}.inp", "w") as f:
        f.write(f"%method\nFunctional {functional}\nBasis {basis_set}\nend\n")
    return f"Input file for ORCA has been generated successfully as {molecule}.inp"

@tool
def orca_executor(execution_command: str) -> str:
    """Executes ORCA software for quantum mechanical calculations using prepared input files."""
    return "ORCA calculations have been completed successfully. The xyz has been optimized and the ground state energy is -100.0 kJ/mol."

@tool
def database_web_search(query: str) -> dict:
    """Queries and retrieves information from external databases or web sources."""
    return "No results found for the given query."

@tool
def visualize_xyz(xyz_file_path:str):
    """
    Visualize a molecule from an XYZ file and show to the user.
    
    Args:
        xyz_file_path (str): Path to the input XYZ file.
    """
    print("Visualizing molecule from XYZ file...")
    # Create molecular graph and read the XYZ file
    mg = MolGraph()
    mg.read_xyz(xyz_file_path)
    
    # Generate interactive 3D visualization
    fig = mg.to_plotly()
    
    # Display the visualization
    fig.show()
    
    # Save the visualization as a PNG image
    #fig.write_image(output_image_path)
    print("Molecule visualization has been displayed successfully.")

    return f"We have seen the molecule from the xyz file. Looks great thank you! ur job is done"


"""# DFT related tools"""
from AgentNet.config import python_ssh_client_mariana

@tool
def send_file_to_hpc(local_file_name:str):
    """
    Send the file from host computer to the HPC working directory for further calculations.
    """
    work_dir = python_ssh_client_mariana.working_directory
    remote_file_path = os.path.join(work_dir, local_file_name)
    python_ssh_client_mariana.transfer_file_to_host(local_file_name,remote_file_path)
    return f"File {local_file_name} has been successfully transferred to the HPC."

@tool
def send_file_to_local(remote_file_name:str):
    """
    Send the file from HPC to the host computer for further analysis.
    """
    local_file_path = remote_file_name
    remote_path = python_ssh_client_mariana.working_directory
    full_path = os.path.join(remote_path, remote_file_name)
    python_ssh_client_mariana.transfer_file_from_host(full_path,local_file_path)
    return f"File {remote_file_name} has been successfully transferred to the local machine."

@tool
def hpc_shell(shell_command:str):
    """
    Execute shell command on the HPC.
    The environment has ORCA installed. 
    The shell is already at working directory
    """
    if "orca" in shell_command:
        human_consent = input("The shell command contains 'orca'. Are you sure you want to proceed? (yes/no): ")
        if human_consent.lower() != "yes":
            return "Shell command execution has been aborted.TERMINATE IMMEDIATELY"
        shell_command = f"/w/185/opt/orca/orca601/{shell_command}"

    output = python_ssh_client_mariana.execute_command(shell_command)
    return f"Shell command {shell_command} has been executed on the HPC, the output is: {output}"


@tool
def read_file_content(file_path):
    """
    Reads and returns the content of a specified file as a string.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        str: The content of the file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return f"Error: The file '{file_path}' does not exist."
    except IOError:
        return f"Error: Could not read the file '{file_path}'."

@tool 
def search_by_keywords(file_path, keywords:Union[str,List[str]]):
    """
    Extracts rows from a file that contain any of the specified keywords.
    Can be used to browse through output files to find the needed information.
    Ex: search_by_keywords("output.txt", ["energy", "optimization"])

    Args:
        file_path (str): Path to the file to search.
        keywords (list of str or str): The keywords to look for.

    Returns:
        list of str: Lines from the file that contain any of the keywords.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    try:
        # Convert keywords to lowercase for case-insensitive matching
        keywords_lower = [keyword.lower() for keyword in keywords]

        matching_lines = []
        with open(file_path, 'r') as file:
            for line in file:
                # Check if any keyword is in the current line
                if any(keyword in line.lower() for keyword in keywords_lower):
                    matching_lines.append(line.strip())  # Strip trailing whitespace

        return matching_lines

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.")
        return []

@tool
def orca_proposal(input_context:str):
    """Propose content of the ORCA input file. When used, provide recommended configuration, xyz file and objective for calculation"""
    from AgentNet.Agent import LangChainChatBot
    chatbot = LangChainChatBot(model="gpt-4o")
    message = chatbot.invoke(f"""You propose ORCA input file, when xyz file is specified, you should simply include the file name, not the coordinates
                                Use at least 4 cores for calculations.
            Example:
            Request: Generate ORCA input file with <functional> <basis set> using <xyz filename> for specific objective

            Answer:
            'orca
            ! <functional> <basis set> <convergence criteria> <other settings>

            CONTENT HERE TO SOLVE THE OBJECTIVE, for example specify xyz coordinate include in output ..etc
                             
            * xyzfile 0 1 <xyz filename>
            '
            Current Request: {input_context}
             
            Your answer: <FILL YOUR ANSWER> """)
    return message.content


if __name__ == "__main__":

   # rdkit_xyz_generation("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    # visualize_xyz("rdkit_generated.xyz")
   # print(read_file_content("test.inp"))
    print(search_by_keywords("test.inp","pal"))

    #xtb_geometry_optimization(r"CN1C=NC2=C1C(=O)N(C(=O)N2C)C.xyz")
    #print(crest_geometry_optimization("caffeine.xyz"))
    #print(xtb_geometry_optimization("caffeine.xyz"))
    #print(functional_selector("caffeine"))
    #print(basis_set_selector("caffeine"))
    #print(input_file_generator("caffeine", "B3LYP", "6-311+G(d,p)"))
    #print(orca_executor("orca caffeine.inp"))
    #print(database_web_search("caffeine"))
    #ssh_client = SSHClient()
    #ssh_client.connect("