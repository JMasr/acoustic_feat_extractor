"""Sets up the click plugin group for packages to plug into."""

from importlib.metadata import entry_points
from pathlib import Path

from clapper.click import AliasedGroup, user_defaults_group
import clapper.logging
from clapper.rc import UserDefaults
import click
from click_plugins import with_plugins
import xdg

logger = clapper.logging.setup("bob")


def legacy_rc_checker(func):
    home = xdg.xdg_config_home()
    old_rc = Path(home) / "../.bobrc"
    new_rc = Path(home) / "bobrc.toml"
    if old_rc.is_file():
        click.echo(
            "You have a legacy .bobrc file. The current configuration file "
            "needs to be located in ~/.config/bobrc.toml. Will now attempt to"
            "copy it to the correct location..."
        )
        if not new_rc.is_file():
            new_rc.write_text(old_rc.read_text())  # Copy old -> new
            click.echo(
                "Your config file was copied to its new location. You can "
                "remove the old file ('~/.bobrc') if you don't use older "
                "(<12.0.0) versions of bob at the same time as this."
            )
        else:
            click.echo(
                "WARNING: You have a legacy ~/.bobrc file but also a new "
                "~/.config/bobrc.toml. The configurations files were not "
                "altered. Consider removing ~/.bobrc if you don't need it."
            )
    return func


# Back-compatibility: ~/.bobrc needs to be moved to ~/.config/bobrc.toml
@legacy_rc_checker
# HACK: entry_points in python <3.10 takes no parameter but returns a dictionary
#     Replace with entry_points(group="bob.cli") when dropping py3.9 support.
@with_plugins(entry_points().select(group="bob.cli"))
@click.group(
    cls=AliasedGroup,
    context_settings={"help_option_names": ("-h", "--help")},
)
def bob_main_cli():
    """The main command line interface for bob."""
    # An entry-point `bob` is created at the package level (pyproject.toml)
    # pointing here.
    #
    # Packages that want to insert their sub-commands in `bob` have to register
    # a unique entry-point in the `bob.cli` group in their pyproject.toml:
    #   [project.entry-points."bob.cli"]
    #       custom = "bob.custom.cli:custom_command"
    # (and install with `pip install -e package_location`)
    #
    # Then, calling `bob custom` will call the `custom_command` function.
    pass


@user_defaults_group(logger=logger, config=UserDefaults("bobrc.toml", logger=logger))
def bob_config(**kwargs):
    "Allows reading and writing into the user configuration."
    pass


bob_main_cli.add_command(bob_config, "config")
