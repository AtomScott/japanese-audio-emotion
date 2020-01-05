import os, sys
import importlib, inspect
import argparse, shutil
from numpydoc.docscrape import FunctionDoc, NumpyDocString, ClassDoc
from inspect import getmembers, isfunction, isclass
from pkgutil import walk_packages
import re

import warnings
warnings.filterwarnings("ignore")

def get_package_contents(package_list):
    """A function to find paths to modules
    
    Parameters
    ----------
    package_list : List
        A list of paths to each package.
        Insert `None` to find all modules in the current working directory.

    Returns
    -------
    List
        A list of module paths, joined by a '.' instead of `os.sep`.
    """
    module_list = ['.'.join([fileFinder.path,modName]) for fileFinder , modName, _ in walk_packages(package_list)]
    return module_list

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-o','--out_dir', default='./docs')
    parser.add_argument('-i','--input_dirs', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-x','--overwrite', action='store_true', default=False)
    parser.add_argument('-p','--prefix', default='./docs')
    parser.add_argument('-s','--suffix', default='')
    args = parser.parse_args()
    return args

def get_module_contents(mod):
    lst = [('.'.join((inspect.getmodule(obj).__name__, name)), obj) for name, obj in getmembers(module) if isfunction(obj) or isclass(obj)]
    return lst

def write_mod_doc(mod, f):
    short_desc = mod.__doc__.split("\n")[0] if mod.__doc__ != None else ' '
   
    f.write('\n'.join([ \
        '---',
        'layout: post',
        'title: {0}'.format(mod.__name__),
        'description: >',
        ' '+short_desc,
        '---', '\n']))

    f.write('# ' +  '{0}'.format(mod.__name__) + '\n')

    if short_desc != ' ':
        f.write('\n'.join([
            '## Description', 
            '{0}'.format(mod.__doc__)]))

    f.write('---\n')

def generate_text(key, item):

    def parse_signature(key, item):
        match = re.match(r"(.*)\((.*)\)" ,item)
        if match:
            na, so = match.groups()
            return '#### **{0}(** *{1}*  **)** {{#signature}}\n'.format(name, so)
        else:
            return '\n'

    def parse_text(item, sep='\n'):
        # if >>> is code block
        # if .. math:: or :math: is math 
        for i, s in enumerate(item):
            if s.startswith('>>>'): 
                item.insert(i, '~~~python')
                item.append('~~~')
                break
            elif s.startswith('.. math::'):
                item[i] = item[i].replace('.. math::', '$$\n')
                while True:
                    i += 1
                    if i >= len(item): break
                    if not item[i].startswith('\t'): break
                item.insert(i, '$$')
            math = re.match(r"(.*):math:`(.*)`(.*)" ,s)
            if math:
                item[i] = '$$'.join(math.groups())
        
        return sep.join(item)

    def parse_section(key, item):
        s = ['##### {0} {{#section}}\n'.format(key), '<dl>']
        for p in item: 
            s.append('<dt markdown=\'1\'>' + '`{0}` : *{1}*'.format(p.name if p.name else " ", p.type) + '\n</dt>')
            s += ["\t<dd markdown=\'1\'> {0} \n</dd>\n".format(''.join(p.desc))]
        s.append('</dl>')
        s.append('')
        return '\n'.join(s)

    def parse_block(key, item):
        s = ['##### {0} {{#block-header}}'.format(key)]
        s.append(parse_text(item))
        return '\n'.join(s)

    def parse_color_block(key, item):
        s = ['##### **{0}**'.format(key)]
        for lst, desc in item:
            names = ', '.join([name for name, _ in lst])
            s.append(': '.join([names, parse_text(desc, ' ')]))
        return '<div class=\'color-block\' markdown=\'1\'>'+'\n'.join(s)+'\n</div>'

    if key == 'Signature':
        return parse_signature(key, item)

    if key in ['Summary', 'Extended Summary']: 
        return parse_text(item)

    if key in ['Parameters', 'Returns', 'Yields', 'Receives', 'Raises', 'Warns', 'Other Parameters']:
        return parse_section(key, item)

    if key in ['Notes', 'Warnings', 'References', 'Examples']:
        return parse_block(key, item)

    if key == 'See Also': 
        return parse_color_block(key, item)

    if key == 'Methods': 
        raise NotImplementedError(key, item)
    if key == 'index': 
        raise NotImplementedError(key, item)

    raise KeyError(key, item)

if __name__ == "__main__":
    # List of paths packages
    args = parse_args()
    package_dirs = args.input_dirs
    out_dir = args.out_dir

    assert args.overwrite == os.path.exists(out_dir), \
        "{0}".format('Path does not exist' if args.overwrite else 'Not given permission to overwrite')
    if args.overwrite: shutil.rmtree(out_dir)        
    
    for module in get_package_contents(package_dirs):
        module = importlib.import_module(module)
        print(module.__name__)

        # path = os.path.join(out_dir, module.__name__.replace('.', os.sep))
        # dir_path, _ = os.path.split(path)

        file_path = os.path.join(out_dir, args.prefix + module.__name__ + args.suffix + '.md')
        os.makedirs(out_dir, exist_ok=True)

        with open(file_path,"w+") as f:
            intro_txt = 'Writing to path {0}'.format(file_path)
            bars = '-' * len(intro_txt)
            print('\t'+bars)
            print('\t'+intro_txt)
            # write module details
            

            write_mod_doc(module, f)
            
            for name, obj in get_module_contents(module):
                # Don't document imported modules!
                if not name.startswith(module.__name__):
                    continue

                print('\t {0}'.format(name))

                if isfunction(obj):
                    doc = FunctionDoc(obj)
                elif isclass(obj):
                    doc = ClassDoc(obj)

                # f.write(str(doc))

                s = []
                for key, item in doc._parsed_data.items():
                    if item: # filter for empty collections and None
                        try: 
                            txt = generate_text(key, item)
                        except NotImplementedError as e:
                            print('\t\t Error: {0} autodoc has not been implemented yet'.format(key))
                        except KeyError as e:
                            print('\t\t Error: {0} is not Numpy style docstring'.format(key))
                        s.append(txt)
                
                s.insert(1,'<div class=\'desc\' markdown="1">')
                s += ['---','</div>']
                f.write('\n'.join(s))
            print('\t'+bars)

            
            

    #     # print(f'\t\t{inspect.getdoc(module)}')

    #     # for fname in file_list:
    #     #     if fname.endswith('.py'):   
    #     #         print(f'\t{fname}')

    #     #         module = importlib.import_module('.'.join([dir_name, fname.replace('.py', '')]))
    #     #         print(dir(module))
    #     #         if 'foo' in dir(module):
    #     #             doc = FunctionDoc(module.foo)
    #     #             print(doc._parsed_data)
    #             # print(inspect.getmembers(module, predicate=inspect.ismethod))

    #             # doc = FunctionDoc(dir(module))
    #             # print(f'\t{doc}')

    #         # doc = FunctionDoc()
    #     # Remove the first entry in the list of sub-directories
    #     # if there are any sub-directories present
    #     # if len(subdirList) > 0:
    #     #     del subdirList[0]