"""
precipfm.cli
============

Implements the command line interface (CLI) of the precipfm package.
"""
from calendar import monthrange
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from concurrent.futures import ProcessPoolExecutor, as_completed

from precipfm.merra import (
    download_dynamic,
    download_static
)

from precipfm import geos, imerg, mrms, merra_precip
from precipfm.obs import gpm, cpcir, patmosx


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("precipfm")
console = Console()


LOGGER = logging.getLogger(__name__)


@click.group()
def precipfm():
    """The command line interface for the precipfm package."""
    pass


def validate_date(ctx, param, value):
    """Validate the year, month, and optionally day arguments."""
    try:
        if param.name == 'day' and value is not None:
            datetime(year=ctx.params['year'], month=ctx.params['month'], day=value)
        elif param.name == 'month':
            datetime(year=ctx.params['year'], month=value, day=1)
        elif param.name == 'year':
            datetime(year=value, month=1, day=1)
        return value
    except ValueError as e:
        raise click.BadParameter(f"Invalid {param.name}: {value}. Error: {str(e)}")


def validate_output_path(ctx, param, value):
    """Validate the output path exists."""
    if not os.path.exists(value):
        raise click.BadParameter(f"The output path does not exist: {value}")
    if not os.path.isdir(value):
        raise click.BadParameter(f"The output path is not a directory: {value}")
    return value


@precipfm.command(name="extract_data")
@click.argument('year', type=int, callback=validate_date)
@click.argument('month', type=int, callback=validate_date)
@click.argument('days', nargs=-1, type=int, required=False, callback=validate_date)
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_data_dynamic(
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract data for a given YEAR, MONTH, and optional DAY, and write output to to OUTPUT_PATH.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        logger.info(f"Extracting data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        logger.info(f"Extracting data for all days in {year}-{month:02d} to {output_path}.")


    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))


    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(download_dynamic, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    download_dynamic(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


@precipfm.command(name="extract_geos_data")
@click.argument('year', type=int, callback=validate_date)
@click.argument('month', type=int, callback=validate_date)
@click.argument('days', nargs=-1, type=int, required=False, callback=validate_date)
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_geos_data(
        year: int,
        month: int,
        days: List[int],
        output_path: Path,
        n_processes: int = 1
) -> None:
    """
    Extract data for a given YEAR, MONTH, and optional DAY, and write output to to OUTPUT_PATH.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        logger.info(f"Extracting data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        logger.info(f"Extracting data for all days in {year}-{month:02d} to {output_path}.")


    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))


    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(geos.download_dynamic, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    geos.download_dynamic(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


@precipfm.command(name="extract_data_static")
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
def extract_data_static(
        output_path: Path,
) -> None:
    """
    Extract static MERRA data and write file to OUTPUT_PATH.
    """
    try:
        download_static(output_path)
    except Exception as e:
        logger.exception(f"Error downloading static data.")


@precipfm.command(name="extract_imerg_data")
@click.argument('year', type=int, callback=validate_date)
@click.argument('month', type=int, callback=validate_date)
@click.argument('days', nargs=-1, type=int, required=False, callback=validate_date)
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_imerg_data(
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract IMERG data for finetuning the PrithviWxC.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        logger.info(f"Extracting IMERG data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        logger.info(f"Extracting IMERG data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(imerg.download, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    imerg.download(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


@precipfm.command(name="extract_merra_precip")
@click.argument('year', type=int, callback=validate_date)
@click.argument('month', type=int, callback=validate_date)
@click.argument('days', nargs=-1, type=int, required=False, callback=validate_date)
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_merra_precip(
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract IMERG data for finetuning the PrithviWxC.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        logger.info(f"Extracting MERRA Precip data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        logger.info(f"Extracting MERRA Precip data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(merra_precip.download, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    merra_precip.download(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)


@precipfm.command(name="extract_mrms_data")
@click.argument('year', type=int, callback=validate_date)
@click.argument('month', type=int, callback=validate_date)
@click.argument('days', nargs=-1, type=int, required=False, callback=validate_date)
@click.argument('output_path', type=click.Path(writable=True), callback=validate_output_path)
@click.option('--n_processes', default=1, type=int, help="Number of processes to use for downloading data.")
def extract_mrms_data(
        year: int,
        month: int,
        days: int,
        output_path: Path,
        n_processes: int
) -> None:
    """
    Extract MRMS data for evaluating the PrithviWxC.

    YEAR and MONTH are required. DAY is optional and defaults to extracting data for
    all days of the month.
    """
    if days:
        logger.info(f"Extracting MRMS data for {year}-{month:02d} on days {', '.join(map(str, days))} to {output_path}.")
    else:
        logger.info(f"Extracting MRMS data for all days in {year}-{month:02d} to {output_path}.")

    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(mrms.download, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    mrms.download(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)

precipfm.command(name="extract_gpm_data")(gpm.process_sensor_data)
precipfm.command(name="extract_cpcir_data")(cpcir.process_data)
precipfm.command(name="extract_patmosx_data")(patmosx.extract_patmosx_data)
precipfm.command(name="extract_geos_forecast_data")(geos.download_geos_forecast)
