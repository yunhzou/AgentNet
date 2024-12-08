from dotenv import load_dotenv
import os
from AgentNet.Database.db import connect_to_mongodb
from AgentNet.Utils import PythonSSHClient
load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")
mongodb_connection_string = os.getenv("MONGODB_URL")
nosql_service = connect_to_mongodb(uri = mongodb_connection_string)
"""SSH Client for HPC"""
python_ssh_client_mariana = PythonSSHClient(
    hostname="mariana.matter.sandbox",
    username="yunhengzou",
    key_file_path=None,  # Path to your private key file
    working_directory="/u/yunhengzou/el-agente/",
    env="el-agente" )

python_ssh_client_mariana.add_configure_command("module load orca/6.0.1")