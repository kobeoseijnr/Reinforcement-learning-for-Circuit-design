import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import shutil
import platform
import tempfile
import yaml
import IPython
debug = False


class AutoCktYAMLLoader(yaml.SafeLoader):
    """YAML loader that knows how to handle python/tuple tags."""


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


AutoCktYAMLLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)


class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.join(tempfile.gettempdir(), "ckt_da")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        self.autockt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=AutoCktYAMLLoader)
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path+'/'+design_netlist
 
        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        self.ngspice_cmd = self._resolve_ngspice_cmd()

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)+str(random.randint(0,10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if '.include' in line:
                regex = re.compile("\.include\s*\"(.*?)\"")
                found = regex.search(line)
                if found:
                    model_rel_path = os.path.join('eval_engines', 'ngspice', 'ngspice_inputs', 'spice_models', '45nm_bulk.txt')
                    path_to_model = os.path.abspath(os.path.join(self.autockt_root, model_rel_path))
                    path_to_model = path_to_model.replace("\\", "/")
                    lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile(r"wrdata\s*([\w\.]+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    replacement = replacement.replace("\\", "/")
                    # wrap path in quotes to handle spaces
                    quoted = f"\"{replacement}\""
                    lines[line_num] = lines[line_num].replace(found.group(1), quoted)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        null_dev = "NUL" if os.name == "nt" else "/dev/null"
        command = "\"{cmd}\" -b \"{netlist}\" > {null} 2>&1".format(
            cmd=self.ngspice_cmd,
            netlist=fpath.replace("\\", "/"),
            null=null_dev
        )
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info

    def _resolve_ngspice_cmd(self):
        """Resolve NGSpice executable path with multiple fallback strategies."""
        # 1. Check environment variable
        env_cmd = os.environ.get("NGSPICE_CMD")
        if env_cmd and os.path.exists(env_cmd):
            return env_cmd
        
        # 2. Check PATH
        for candidate in ("ngspice", "ngspice.exe"):
            path = shutil.which(candidate)
            if path and os.path.exists(path):
                return path
        
        # 3. Check common Windows installation paths
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Spice64\bin\ngspice.exe",
                r"C:\Program Files (x86)\Spice64\bin\ngspice.exe",
                r"C:\Spice64\bin\ngspice.exe",
                r"C:\Program Files\ngspice\bin\ngspice.exe",
                r"C:\ngspice\bin\ngspice.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Spice64\bin\ngspice.exe"),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        # 4. Check common Linux/Mac paths
        else:
            common_paths = [
                "/usr/bin/ngspice",
                "/usr/local/bin/ngspice",
                "/opt/ngspice/bin/ngspice",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        # 5. Last resort: try to find in current directory or parent
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_paths = [
            os.path.join(script_dir, "ngspice.exe"),
            os.path.join(script_dir, "ngspice"),
            os.path.join(os.path.dirname(script_dir), "ngspice.exe"),
            os.path.join(os.path.dirname(script_dir), "ngspice"),
        ]
        for path in local_paths:
            if os.path.exists(path) and os.path.isfile(path):  # Ensure it's a file, not directory
                return path
        
        # If still not found, provide helpful error with instructions
        error_msg = (
            "NGSpice executable not found.\n"
            "Please do ONE of the following:\n"
            "1. Install NGSpice from https://ngspice.sourceforge.net/download.html\n"
            "2. Add NGSpice to your system PATH\n"
            "3. Set environment variable: setx NGSPICE_CMD \"C:\\path\\to\\ngspice.exe\"\n"
            "4. Or create a file 'ngspice_path.txt' in this directory with the full path to ngspice.exe"
        )
        
        # Check for ngspice_path.txt as final fallback
        config_file = os.path.join(os.path.dirname(__file__), "ngspice_path.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    path = f.read().strip().strip('"')
                    if os.path.exists(path):
                        return path
            except:
                pass
        
        raise RuntimeError(error_msg)


    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
