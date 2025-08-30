# Create a comprehensive household energy consumption ML project structure
import os
import datetime

# Define project structure
project_structure = {
    'household_energy_consumption_ml': {
        'data': {
            'raw': {},
            'processed': {}
        },
        'src': {
            'data_processing': {},
            'models': {},
            'evaluation': {},
            'utils': {}
        },
        'notebooks': {},
        'results': {
            'models': {},
            'plots': {},
            'reports': {}
        },
        'config': {},
        'tests': {}
    }
}

# Create directory structure
def create_directories(structure, base_path=''):
    for name, content in structure.items():
        path = os.path.join(base_path, name) if base_path else name
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_directories(content, path)
        else:
            os.makedirs(path, exist_ok=True)

create_directories(project_structure)
print("Project directory structure created successfully!")