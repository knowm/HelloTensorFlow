## Compile TF from source

1. Install XCode via AppStore.
1. [Bazel](https://github.com/bazelbuild/bazel) is a software dependency and build tool similar to ANT and Maven. Installation Instructions are [here](https://docs.bazel.build/versions/master/install-os-x.html#install-on-mac-os-x-homebrew).

```
brew install bazel
bazel version
brew upgrade bazel
pip3 install six numpy wheel 
brew install coreutils 
```

1. `cd path/to/tensorflow home/`
1. `git clone git@github.com:knowm/tensorflow.git` Change Knowm's tensorflow fork to whatever fork you need.
1. We need a custom `build_tf.sh` because to add the CPU optimization flags as well as to combine a few separate related commands. The source of our build file is [here](https://gist.github.com/venik/9ba962c8b301b0e21f99884cbd35082f). This retrieves all CPU features and applies some of them to build TF, which makes TF faster as it will utilize specialized CPU instructions if your computer has them.
1. Some more tools setup:

```
bazel clean --expunge
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license
bazel clean --expunge
```

Here is our `build_tf.sh`:

```bash
#!/bin/bash

# Author: Sasha Nikiforov

# source of inspiration
# https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions

raw_cpu_flags=`sysctl -a | grep machdep.cpu.features | cut -d ":" -f 2 | tr '[:upper:]' '[:lower:]'`
COPT="--copt=-march=native"

for cpu_feature in $raw_cpu_flags
do
    case "$cpu_feature" in
        "sse4.1" | "sse4.2" | "ssse3" | "fma" | "cx16" | "popcnt" | "maes")
            COPT+=" --copt=-m$cpu_feature"
        ;;
        "avx1.0")
            COPT+=" --copt=-mavx"
        ;;
        *)
            # noop
        ;;
    esac
done

bazel clean
./configure
bazel build -c opt $COPT -k //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip3 install --upgrade /tmp/tensorflow_pkg/`ls /tmp/tensorflow_pkg/ | grep tensorflow`
```

Notice the line: `#./configure`, which you'll need to run only the first time to configure tensorflow. In the future, just comment it out.

If you cloned Knowm's tensorflow fork, this file will be in the root of the `tensrflow` directory.

1. Move to `tensorflow` directory
1. Make sure the file is executable

```bash
cd path/to/tensorflow
chmod +x build_tf.sh
```

To run the script run:

```
./build_tf.sh
```
Path to python, when asked, is: `/usr/local/bin/python3`.

The first build might take over two hours!



### Getting Tensorboard to also work after building from Source

This references an issue I opened: <https://github.com/tensorflow/tensorboard/issues/812>

```
pip3 install tb-nightly
Collecting tb-nightly
  Downloading tb_nightly-1.5.0a20171213-py3-none-any.whl (3.0MB)
    100% |████████████████████████████████| 3.0MB 425kB/s 
Requirement already satisfied: numpy>=1.12.0 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: html5lib==0.9999999 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: werkzeug>=0.11.10 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: protobuf>=3.4.0 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: bleach==1.5.0 in /usr/local/lib/python3.6/site-packages (from tb-nightly)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/site-packages (from protobuf>=3.4.0->tb-nightly)
Installing collected packages: tb-nightly
Successfully installed tb-nightly-1.5.0a20171213
```

## Re-run an example

```python
python3 tensorflow/examples/tutorials/mnist/mnist_softmax.py
```

Congratulations!!

## How Tensorflow is Structured


`core/` contains the main C++ code and runtimes.

`ore/ops/` contains the "signatures" of the operations
`core/kernels/` contains the "implementations" of the operations (including CPU and CUDA kernels)
`core/framework/` contains the main abstract graph computation and other useful libraries
`core/platform/` contains code that abstracts away the platform and other imported libraries (protobuf, etc)

TensorFlow relies heavily on the Eigen library for both CPU and GPU calculations.  Though some GPU kernels are implemented directly with CUDA code.

bazel builds certain C++ code using gcc/clang, and certain CUDA code (files with extension .cu.cc) with nvcc.

`python/ops/` contain the core python interface
`python/kernel_tests/` contain the unit tests and lots of example code
`python/framework/` contains the python abstractions of graph, etc, a lot of which get serialized down to proto and/or get passed to swigged session calls.
`python/platform/` is similar to the C++ platform, adding lightweight wrappers for python I/O, unit testing, etc.

`contrib/*/` directories generally mimic the root tensorflow path (i.e., they have core/ops/, etc)

### Bazel Unit tests

`bazel run <target>` lets you run a unit test (bazel test will log the output of one or more tests to a file rather than printing it directly). So for example if you're modifying the one hot op (tensorflow/core/kernels/one_hot_op.cc), you might want to run the Python unit test suit for that op (tensorflow/python/kernel_tests/one_hot_op_test.py):

`bazel run //tensorflow/python/kernel_tests:one_hot_op_test`

Or go to the directory and use a relative build target:

```
cd python/kernel_tests
bazel run :one_hot_op_test
```

Test targets are defined in BUILD files. Any changes to dependencies should be picked up automatically. After the first run, compilation should be quite snappy, as most of it will be cached.

If you're developing changes which will eventually be integrated into software which uses TensorFlow, it's likely going to be a more pleasant experience if you first unit test the TensorFlow changes before moving on to integration testing (where you'll e.g. bundle up TensorFlow into a pip package).

### Building with Bazel Overview

Bazel is used for building.

[Bazel](http://bazel.io) is a build tool just like other build tools like [cmake](http://cmake.org) and [make](https://www.gnu.org/software/make/). The steps you listed is the correct way to get updates from master. The build step could take long the first time you build TensorFlow. Later builds, after updates from master, should be faster, as Bazel, just like any other build tool, doesn't re-build targets whose dependencies have not been modified.

[--source](https://docs.bazel.build/versions/master/build-ref.html)

#### BUILD File

Bazel builds software from source code organized in a directory called a workspace. Source files in the workspace are organized in a nested hierarchy of packages, where each package is a directory that contains a set of related source files and one **BUILD** file. The BUILD file specifies what software outputs can be built from the source.


#### Packages


A package is defined as a directory containing a file named BUILD, residing beneath the top-level directory in the workspace. A package includes all files in its directory, plus all subdirectories beneath it, except those which themselves contain a BUILD file.

For example, in the following directory tree there are two packages, my/app, and the subpackage my/app/tests. Note that my/app/data is not a package, but a directory belonging to package my/app.

```
src/my/app/BUILD
src/my/app/app.cc
src/my/app/data/input.txt
src/my/app/tests/BUILD
src/my/app/tests/test.cc
```

#### Workspace

A workspace is a directory on your file system that contains the source files for the software you want to build, as well as symbolic links to directories that contain the build outputs. Each workspace directory has a text file named WORKSPACE which may be empty, or may contain references to external dependencies required to build the outputs.

### Targets

A package is a container. The elements of a package are called targets. Most targets are one of two principal kinds, files and rules. Additionally, there is another kind of target, package groups, but they are far less numerous.

Files are further divided into two kinds. Source files are usually written by the efforts of people, and checked in to the repository. Generated files, sometimes called derived files, are not checked in, but are generated by the build tool from source files according to specific rules.

The second kind of target is the rule. A rule specifies the relationship between a set of input and a set of output files, including the necessary steps to derive the outputs from the inputs. The outputs of a rule are always generated files. The inputs to a rule may be source files, but they may be generated files also; consequently, outputs of one rule may be the inputs to another, allowing long chains of rules to be constructed.

### C++/Python Interface

[Swig](https://en.wikipedia.org/wiki/SWIG) is used to interface c++ function to python.

### GRPC

[GRPC](https://github.com/grpc/grpc), a high performance, open-source universal RPC framework, is used for distributed computing.

## Incrementally Building Tensorflow

For developing Tensorflow, we obviously don't want to have to build the entire source tree from scratch every single time a change is made. Therefore, we want to build TF *incrementally* each time a change is made.

To accomplish this, we need to first understand what our build file `build_tf.sh` does, in particular the last few steps:

```
bazel clean
./configure
bazel build -c opt $COPT -k //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip3 install --upgrade /tmp/tensorflow_pkg/`ls /tmp/tensorflow_pkg/ | grep tensorflow`
```


#### bazel clean

This deletes all the build artifacts.


#### ./configure

This causes the GUI to ask the user a list of questions to configure the build.


#### bazel build -c opt $COPT -k //tensorflow/tools/pip_package:build_pip_package

This does the compiling. It is smart enough to know to only build source that has been modified since the last build.


#### bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

This creates a python wheel file in `/tmp/tensorflow_pkg`


#### pip3 install --upgrade /tmp/tensorflow_pkg/ls /tmp/tensorflow_pkg/ | grep tensorflow


This takes the wheel file and makes it an executable on the system via calling `tensorflow`.

For an incremental build we only want to run the last 3 (of 5) lines of the build file and to do so we can just comment out the first 2 lines of code. The incremental build then takes less than 2 minutes. 


## Hacking Tensorflow at the Python Level

Let's see if we can change some code in TF itself in a trivial way, recompile, re-run a `hellotf.py` program and verify that our incremental build setup is working as expected. We'll use the following simple TF app:

```
# https://mubaris.com/2017-10-21/tensorflow-101

# Import TensorFlow
import tensorflow as tf

# Define Constant
output = tf.constant("Hello, World")

# To print the value of constant you need to start a session.
sess = tf.Session()

# Print
print(sess.run(output))

# Close the session
sess.close()
```

, which produces:

```
b'Hello, World'
```

What if we change the `tf.constant` code to append the String `_hack` onto the end of the inputted constant?

In `constant_op.py`, line 212, we can modify it like this:

```
value+'_hack', dtype=dtype, shape=shape, verify_shape=verify_shape))
```

After recompiling and re-running, we get:

```
b'Hello, World_hack'

```

, which confirms that our incremental build setup is indeed working!
