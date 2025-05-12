import configargparse

def get_config():
    parser = configargparse.ArgParser(default_config_files=['config.yaml'],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Path to config file')
    parser.add('--music_dir', type=str, default='music', help='Directory with original MP3 music files')
    parser.add('--wav_output_dir', type=str, default='musicWav', help='Directory to save converted WAV files')
    parser.add('--n_fft', type=int, default=4096)
    parser.add('--hop_length', type=int, default=512)
    parser.add('--bands', nargs='+', type=int, default=[0, 10, 40, 80, 120, 180, 300, 500, 800, 1200])
    parser.add('--fan_value', type=int, default=10)
    parser.add('--max_time_delta', type=float, default=5.0)

    parser.add('--snippet', type=str, default='snippets/perfect.wav', help="snippet file to analyze")
    parser.add('--highlight', type=str, default="Unknown Song Name", help="Song to highlight in plot")
    parser.add('--offset', action='store_true', help="Use offset-based matching")
    parser.add('--noise', action='store_true', help="Add noise to audio before matching")
    
    return parser.parse_args(args=[])