import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams


def generate_synthetic_data(length: int = 2200, periods: int = 5):
    """The function generates a synthetic one-dimensional array without omissions

    Args:
        length: the length of the array
        periods: the number of periods in the sine wave

    Returns:
        array: an array without gaps
    """

    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise
    return synthetic_data


def get_time_series(len_forecast=5, length=80):
    """ Function returns time series for time series forecasting task """
    synthetic_ts = generate_synthetic_data(length=length)

    X = synthetic_ts[:-len_forecast]
    y = synthetic_ts[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_data = InputData(
        idx=np.arange(len(X)),
        features=np.ravel(X),
        target=np.ravel(X),
        task=task,
        data_type=DataTypesEnum.ts
    )

    start_forecast = len(train_data.features)
    end_forecast = start_forecast + len_forecast
    test_data = InputData(
        idx=np.arange(start_forecast, end_forecast),
        features=np.ravel(X),
        target=np.ravel(y),
        task=task,
        data_type=DataTypesEnum.ts
    )

    return train_data, test_data,