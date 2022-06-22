from cyp.models.cnn_lstm import CNN_LSTM_Model
from cyp.models.cnn_lstm_scratch import CNN_LSTM_scratchModel
from cyp.models.cnn_sun import CNN_Sun_Model
import torch
from pathlib import Path

from cyp.data import MODISExporter, DataCleaner, Engineer
from cyp.models import ConvModel, RNNModel, CNN_LSTM_Model

import fire


class RunTask:
    @staticmethod
    def export(
        export_limit=None,
        major_states_only=True,
        check_if_done=True,
        download_folder=None,
        yield_data_path="data/yield_data.csv",
    ):
        yield_data_path = Path(yield_data_path)
        exporter = MODISExporter(locations_filepath=yield_data_path)
        exporter.export_all(
            export_limit, major_states_only, check_if_done, download_folder
        )

    @staticmethod
    def process(
        mask_path="data/crop_yield-data_mask",
        temperature_path="data/crop_yield-data_temperature",
        image_path="data/crop_yield-data_image",
        yield_data_path="data/yield_data.csv",
        cleaned_data_path="data/img_output",
        multiprocessing=False,
        processes=4,
        parallelism=6,
        delete_when_done=False,
        num_years=14,
        checkpoint=True,
    ):
        mask_path = Path(mask_path)
        temperature_path = Path(temperature_path)
        image_path = Path(image_path)
        yield_data_path = Path(yield_data_path)
        cleaned_data_path = Path(cleaned_data_path)

        cleaner = DataCleaner(
            mask_path,
            temperature_path,
            image_path,
            yield_data_path,
            savedir=cleaned_data_path,
            multiprocessing=multiprocessing,
            processes=processes,
            parallelism=parallelism,
        )
        cleaner.process(
            delete_when_done=delete_when_done,
            num_years=num_years,
            checkpoint=checkpoint,
        )

    @staticmethod
    def engineer(
        cleaned_data_path="data/img_output",
        yield_data_path="data/yield_data.csv",
        county_data_path="data/county_data.csv",
        num_bins=32,
        max_bin_val=4999,
    ):

        cleaned_data_path = Path(cleaned_data_path)
        yield_data_path = Path(yield_data_path)
        county_data_path = Path(county_data_path)

        engineer = Engineer(cleaned_data_path, yield_data_path, county_data_path)
        engineer.process(
            num_bands=9,
            generate="histogram",
            num_bins=num_bins,
            max_bin_val=max_bin_val,
            channels_first=True,
        )

    @staticmethod
    def train_cnn(
        cleaned_data_path=Path("data/img_output"),
        dropout=0.5,
        dense_features=None,
        savedir=Path("data/models"),
        times=[18],
        pred_years=[2017, 2018, 2019, 2020, 2021],
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=1,
        l1_weight=0,
        patience=10,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        histogram_path = Path(cleaned_data_path) / "more_data_histogramfull.npz"

        model = ConvModel(
            in_channels=11,
            dropout=dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def train_rnn(
        cleaned_data_path="data/img_output",
        num_bins=32,
        hidden_size=128,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        times="all",
        pred_years=[2017, 2018, 2019, 2020],
        num_runs=2,
        train_steps=10000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        histogram_path = Path(cleaned_data_path) / "histogram_daymet_all_full.npz"

        model = RNNModel(
            in_channels=11,
            num_bins=num_bins,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def train_cnn_lstm(
        cleaned_data_path="data/img_output",
        num_bins=32,
        hidden_size=128,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        times="all",
        pred_years=[2016],
        num_runs=5,
        train_steps=10000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        histogram_path = Path(cleaned_data_path) / "real_histogram_daymet_all_full.npz"

        model = CNN_LSTM_Model(
            in_channels=11,
            num_bins=num_bins,
            hidden_size=hidden_size,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def train_cnn_lstm_scratch(
        cleaned_data_path="data/img_output",
        num_bins=32,
        hidden_size=256,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        times=[18],
        pred_years=[2017,2018,2019,2020,2021],
        num_runs=5,
        train_steps=10000,
        batch_size=16,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        use_gp=False,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        histogram_path = Path(cleaned_data_path) / "more_data_histogramfull.npz"


        for run in range(num_runs):
            for year in pred_years:
                model = CNN_LSTM_scratchModel(
                    in_channels=11,
                    num_bins=num_bins,
                    hidden_size=hidden_size,
                    dense_features=dense_features,
                    savedir=savedir,
                    use_gp=use_gp,
                    sigma=sigma,
                    r_loc=r_loc,
                    r_year=r_year,
                    sigma_e=sigma_e,
                    sigma_b=sigma_b,
                    device=device,
                )
                model.run(
                    histogram_path,
                    times,
                    [year],
                    1,
                    train_steps,
                    batch_size,
                    starter_learning_rate,
                    weight_decay,
                    l1_weight,
                    patience,
                )

    
    @staticmethod
    def train_cnn_sun(
        cleaned_data_path="data/img_output",
        num_bins=32,
        hidden_size=256,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        times=[34],
        pred_years=[2017,2018,2019,2020,2021],
        num_runs=2,
        train_steps=10000,
        batch_size=16,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        use_gp=False,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        histogram_path = Path(cleaned_data_path) / "more_data_histogramfull.npz"


        for run in range(num_runs):
            for year in pred_years:
                model = CNN_Sun_Model(
                    in_channels=11,
                    num_bins=num_bins,
                    hidden_size=hidden_size,
                    dense_features=dense_features,
                    savedir=savedir,
                    use_gp=use_gp,
                    sigma=sigma,
                    r_loc=r_loc,
                    r_year=r_year,
                    sigma_e=sigma_e,
                    sigma_b=sigma_b,
                    device=device,
                )
                model.run(
                    histogram_path,
                    times,
                    [year],
                    1,
                    train_steps,
                    batch_size,
                    starter_learning_rate,
                    weight_decay,
                    l1_weight,
                    patience,
                )




if __name__ == "__main__":
    fire.Fire(RunTask)
