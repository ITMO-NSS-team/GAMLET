from components.datasets_loader import OpenmlLoader


def main():
    dataset_names = [
        'nomao', 'sylvine', 'kc1', 'jungle_chess_2pcs_raw_endgame_complete', 'credit-g', 'delta_ailerons', 'pol'
    ]
    loader = OpenmlLoader(dataset_names)
    datasets = loader()
    print(f'Datasets "{", ".join(dataset_names)}" are available at the paths:')
    print('\n'.join(str(d) for d in datasets))


if __name__ == '__main__':
    main()
