#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system module containing atom and param attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import re
import sys
import os
import ast
from typing import List, Tuple, Dict, Union


class DecompCls(object):
        """
        this class contains all system attributes
        """
        def __init__(self) -> None:
                """
                init molecule attributes
                """
                # set defaults
                self.atom: Union[List[str], str] = ''
                self.param: Dict[str, Union[str, bool, float]] = {'basis': 'sto-3g', 'loc': 'pm', 'pop': 'mulliken', \
                                                                  'dft': False, 'cube': False, 'thres': .95}


def set_param(param: Dict[str, Union[str, bool]]) -> Tuple[str, Dict[str, Union[str, bool]]]:
        """
        this function sets system parameter attributes from input file
        """
        # read input file
        try:
            with open(os.getcwd()+'/input') as f:
                content = f.readlines()
                for i in range(len(content)):
                    if content[i].strip():
                        if content[i].split()[0][0] == '#':
                            continue
                        elif re.split('=',content[i])[0].strip() == 'atom':
                            atom = ''
                            for j in range(i+1, len(content)):
                                if content[j][:3] == "'''" or content[j][:3] == '"""':
                                    break
                                else:
                                    atom += content[j]
                        elif re.split('=',content[i])[0].strip() == 'param':
                            try:
                                inp = ast.literal_eval(re.split('=',content[i])[1].strip())
                            except ValueError:
                                raise ValueError('wrong input -- error in reading in param dictionary')
                            # update system
                            param = {**param, **inp}
        except IOError:
            sys.stderr.write('\nIOError : input file not found\n\n')
            raise

        return atom, param


if __name__ == "__main__":
    import doctest
    doctest.testmod()


