"""Build the maintenance implementation and run real C++ reference tests.
Python 3 stdlib only. No package downloads, SDK probing or shell execution.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX'))
    parser.add_argument('--target', help='Optional target triple, supported by Zig/Clang')
    parser.add_argument('--force-scalar', action='store_true')
    parser.add_argument('--openmp', action='store_true', help='Requires compiler OpenMP headers/runtime')
    parser.add_argument('--sanitize', action='store_true', help='GCC/Clang address+undefined sanitizers')
    parser.add_argument('--compile-only', action='store_true', help='Never execute a cross-compiled binary')
    args = parser.parse_args()
    compiler = args.cxx or next((shutil.which(c) for c in ('c++', 'g++', 'clang++', 'cl', 'zig')
                                 if shutil.which(c)), None)
    if not compiler:
        parser.error('No C++ compiler found. Install/configure one or pass --cxx PATH; no tests executed.')
    compiler = shutil.which(compiler) or compiler
    name = Path(compiler).stem.lower()
    msvc = name in ('cl', 'clang-cl')
    zig = name == 'zig'
    prefix = [compiler] + (['c++'] if zig else [])
    if args.target:
        if zig:
            prefix += ['-target', args.target]
        elif name.startswith('clang') and not msvc:
            prefix += ['--target=' + args.target]
        else:
            parser.error('--target is supported only with Zig or Clang')
    elif zig and os.name == 'nt':
        # Avoid depending on a separately installed Microsoft SDK.
        prefix += ['-target', 'x86_64-windows-gnu']
    if msvc:
        flags = ['/nologo', '/std:c++17', '/O2', '/W4', '/EHsc', '/Iinclude']
        if args.force_scalar:
            flags += ['/DPAC_FORCE_SCALAR=1']
        if args.openmp:
            flags += ['/openmp']
        if args.sanitize:
            parser.error('--sanitize currently supports GCC/Clang GNU-style drivers only')
    else:
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Iinclude']
        if args.force_scalar:
            flags += ['-DPAC_FORCE_SCALAR=1']
        if args.openmp:
            flags += ['-fopenmp']
        if args.sanitize:
            flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    print('MAINTENANCE_SOURCE_SHA256', hashlib.sha256((ROOT / 'dgemm_final.cpp').read_bytes()).hexdigest())
    suffix = ('-scalar' if args.force_scalar else '') + ('-omp' if args.openmp else '')
    suffix += '-sanitized' if args.sanitize else ''
    suffix += '.exe' if os.name == 'nt' else ''
    targets = [('tests/correctness.cpp', build / ('pac-tests' + suffix)),
               ('bench/benchmark.cpp', build / ('pac-bench' + suffix))]
    for source, output in targets:
        command = prefix + flags + ['dgemm_final.cpp', source]
        command += ['/Fe' + str(output), '/Fo' + str(build) + os.sep] if msvc else ['-o', str(output)]
        print('BUILD', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if args.compile_only:
        print('COMPILE ONLY: no target binaries executed; no correctness or performance claim.')
        return
    subprocess.run([str(targets[0][1])], cwd=ROOT, check=True)
    subprocess.run([str(targets[1][1]), '--m', '13', '--n', '7', '--k', '9',
                    '--rounds', '3', '--warmup', '1'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
