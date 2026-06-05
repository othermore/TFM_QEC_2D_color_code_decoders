from simulator import run_cli_pipeline

if __name__ == "__main__":
    run_cli_pipeline(
        decoder_name='correlated',
        file_prefix='s7_correlated',
        plot_title='Correlated MWPM Decoder'
    )
