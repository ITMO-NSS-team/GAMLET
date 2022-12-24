from pathlib import Path

from meta_automl.data_preparation.data_manager import DataManager


class TestDataManager(DataManager):
    @classmethod
    def get_data_dir(cls) -> Path:
        return cls.get_project_root().joinpath('test/data')
