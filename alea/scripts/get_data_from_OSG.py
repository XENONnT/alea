import os
import sys
import argparse
import glob
import pkg_resources
import subprocess


def execute_remote_command(user,
                           remote_machine,
                           remote_command,
                           skip_command_check=False):
    """Docstring for function.
    This function executes the command

    :arg1: user - user which is used to execute the remote command
    :arg2: remote_machine - remote machine where the command should be executed
    :arg3: remote_command - command which is executed on the remote machine

    :returns: None
    """
    # Ports are handled in ~/.ssh/config since we use OpenSSH
    command = ["ssh", "%s@%s" % (user, remote_machine), remote_command]
    print(" ".join(command))
    ssh = subprocess.Popen(
        command,
        #  bufsize=0,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        #  close_fds=True
    )
    std, err = ssh.communicate()

    if skip_command_check:
        print("No remote check.")
    else:
        print("Checking remote result... Consider '--skip_command_check'")
        if std == []:
            #  error = ssh.stderr.readlines()
            print(err, "ERROR: %s" % err)
        else:
            print("Command executed successfully!")


def parse_args(args):
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser("debug toy submission")
    parser.add_argument("--OSG_path",
                        type=str,
                        required=True,
                        help="Path to copy from OSG")
    parser.add_argument("--midway_target",
                        default="/home/twolf/scratch-midway2/toyoutput_SR0",
                        type=str,
                        help="Target path on midway")
    parser.add_argument("--user",
                        type=str,
                        required=True,
                        help="User used for copy")
    parser.add_argument("--remote_machine",
                        type=str,
                        default="login.xenon.ci-connect.net",
                        help="URL of remote machine")
    parser.add_argument("--skip_command_check",
                        action='store_true',
                        help="Skip checking of remote command.")
    parser.add_argument("--get",
                        action='store_true',
                        help="Execute unpacking and transfer")
    parser.add_argument("--no_unpack",
                        action='store_true',
                        help="Avoid unpacking remotely")

    args = parser.parse_args(args)
    if not args.OSG_path.endswith("/"):
        args.OSG_path += "/"

    return vars(args)


def main(args):
    parsed_args = parse_args(args=args)
    midway_target = parsed_args["midway_target"]
    OSG_path = parsed_args["OSG_path"]
    user = parsed_args["user"]
    remote_machine = parsed_args["remote_machine"]

    # setting up paths
    splitted_path = OSG_path.split("/")
    if splitted_path[-1] == "":
        dirname = splitted_path[-2]
    else:
        dirname = splitted_path[-1]
    full_path_midway = os.path.join(midway_target, dirname)
    if not os.path.exists(full_path_midway):
        os.makedirs(full_path_midway)
    if not full_path_midway.endswith("/"):
        full_path_midway += "/"

    if  parsed_args["no_unpack"]:

        print("No unpacking remotely")
    else:
        # unpack tarballs remotely
        unpacking_script = pkg_resources.resource_filename(
            "binference", "scripts/unpack_data.sh")
        print(unpacking_script)
        cmd = "scp {unpacking_script} {user}@{remote_machine}:~/".format(
            unpacking_script=unpacking_script,
            user=user,
            remote_machine=remote_machine)
        if parsed_args["get"]:
            os.system(cmd)
        else:
            print(cmd)

        remote_command = "source $HOME/{unpacking_script} {OSG_path}".format(
            unpacking_script=os.path.basename(unpacking_script), OSG_path=OSG_path)
        if parsed_args["get"]:
            print("Have a little patience... I am unpacking data remotely...")
            execute_remote_command(
                user=user,
                remote_machine=remote_machine,
                remote_command=remote_command,
                skip_command_check=parsed_args["skip_command_check"])
        else:
            print("REMOTE COMMAND:", remote_command)

    # copy output from remote machine
    pattern = os.path.join(OSG_path, "*.hdf5")
    copy_cmds = []
    copy_cmd = "rsync -a --progress --exclude jobs/ --exclude *tar.gz {user}@{remote_machine}:{OSG_path} {full_path_midway}".format(
        user=user,
        remote_machine=remote_machine,
        full_path_midway=full_path_midway,
        OSG_path=OSG_path
        )
    #  copy_cmd = "scp {user}@{remote_machine}:{pattern} {full_path_midway}".format(
    #      user=user,
    #      remote_machine=remote_machine,
    #      pattern=pattern,
    #      full_path_midway=full_path_midway)
    copy_cmds.append(copy_cmd)

    pattern = os.path.join(OSG_path, "*.yaml")
    copy_cmd = "scp {user}@{remote_machine}:{pattern} {full_path_midway}".format(
        user=user,
        remote_machine=remote_machine,
        pattern=pattern,
        full_path_midway=full_path_midway)
    copy_cmds.append(copy_cmd)

    for cmd in copy_cmds:
        print(cmd)
        if parsed_args["get"]:
            os.system(cmd)

    if parsed_args["get"]:
        print()
        print("Midway path:")
        print(full_path_midway)
        yaml_files = glob.glob(os.path.join(full_path_midway, "*.yaml"))
        print()
        print("------------------")
        print("YAML files:")
        for yaml_file in yaml_files:
            print(os.path.basename(yaml_file))


if __name__ == "__main__":
    main(sys.argv[1:])
