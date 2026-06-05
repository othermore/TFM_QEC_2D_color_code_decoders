from simulator import run_cli_pipeline

if __name__ == "__main__":
    run_cli_pipeline(
        decoder_name='projection',
        file_prefix='s3_projection',
        plot_title='Projection Decoder'
    )
