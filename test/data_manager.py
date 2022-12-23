from pathlib import Path

from meta_automl.data_preparation.data_directory_manager import DataDirectoryManager


class TestDataManager(DataDirectoryManager):
    @classmethod
    def get_data_dir(cls) -> Path:
        return cls.get_project_root().joinpath('test/data')
