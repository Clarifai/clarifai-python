"""
The clarifai-cli as a swiss-knife of ML lifecycle (potentially),
serving purposes from Profile configuration e.g PAT in a $HOME/.clarifi-config.json file,
to later on support multiple purposes like data upload, model CRUD, workflow CRUD etc.
"""
import os
import argparse

from clarifai.actions.list_of_actions import Actions, ActionFactory


def cli():
    parser = argparse.ArgumentParser(add_help=False)
    # main entrypoint for furthering the command
    parser.add_argument("action", help="choose an action from a list of allowed action w.r.t the cli", choices=[el.value for el in Actions])
    # custom helper function
    parser.add_argument("-h", "--help", action='store_true', default=argparse.SUPPRESS,
                        help='general or action/command- specific help', required=False)

    # args associated with "configure" command
    parser.add_argument("--profile", default="default",
                        help='general or action/command- specific help', required=False)

    args = parser.parse_args()
    cli_action_obj = ActionFactory()
    cli_action_obj.execute(args)


if __name__ == "__main__":
    os.system("clarifai-cli configure")
