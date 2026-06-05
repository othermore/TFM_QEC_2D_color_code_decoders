from simulator import run_cli_pipeline

if __name__ == "__main__":
    run_cli_pipeline(
        decoder_name='concat_mwpm',
        file_prefix='s6_concatenated',
        plot_title='Concatenated MWPM Decoder'
    )
