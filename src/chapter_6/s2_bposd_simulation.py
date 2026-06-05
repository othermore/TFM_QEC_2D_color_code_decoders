from simulator import run_cli_pipeline

if __name__ == "__main__":
    run_cli_pipeline(
        decoder_name='bposd',
        file_prefix='s2_bposd',
        plot_title='BP+OSD'
    )
