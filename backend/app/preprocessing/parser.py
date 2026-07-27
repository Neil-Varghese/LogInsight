from pathlib import Path

from logparser.Drain import LogParser


class HDFSParser:
    """
    Parse HDFS logs using LogPai Drain.

    Development:
        max_lines=1000

    Production:
        max_lines=None
    """

    def __init__(
        self,
        depth=4,
        st=0.5,
        log_format="<Date> <Time> <Pid> <Level> <Component>: <Content>",
    ):
        self.depth = depth
        self.st = st
        self.log_format = log_format

    def parse(
        self,
        input_dir,
        output_dir,
        log_file="HDFS.log",
        max_lines=None,
        force=False,
    ):

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Create a small sample file
        # -----------------------------
        parse_file = log_file

        if max_lines is not None:

            sample_name = f"sample_{max_lines}.log"
            sample_path = output_dir / sample_name

            if force or not sample_path.exists():

                print(f"Creating {sample_name}...")

                with open(input_dir / log_file, "r", encoding="utf-8") as fin, \
                     open(sample_path, "w", encoding="utf-8") as fout:

                    for i, line in enumerate(fin):
                        if i >= max_lines:
                            break
                        fout.write(line)

            parse_file = sample_name
            input_dir = output_dir

        structured_file = output_dir / f"{parse_file}_structured.csv"

        if structured_file.exists() and not force:
            print("Structured file already exists.")
            return structured_file

        parser = LogParser(
            log_format=self.log_format,
            indir=str(input_dir),
            outdir=str(output_dir),
            depth=self.depth,
            st=self.st,
        )

        parser.parse(parse_file)

        return structured_file