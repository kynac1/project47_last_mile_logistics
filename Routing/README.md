# To get these examples working:

1. Install Julia: https://julialang.org/downloads/

2. Install the vscode extension for julia

3. Open this directory in vscode and start julia
This can be done using the vscode command `julia: start REPL`, which is accessed from the ctrl+shift+p dropdown. Or you could just run julia directly in the terminal, using either `julia` or the absolute path to the julia executable. I'm not sure which is easier on mac.

4. In the terminal, type `]` to enter the package editing mode.
Run `activate .`
This activates julias version of a conda environment
Then run `instantiate`
This should download all the dependencies I have installed here. To see what they are, look in the Project.toml file.

You should now be able to run openstreetmap.jl under examples. To do this you could include it using include("examples/openstreetmap.jl"). This may take a while to start up.

You can also run it using vscode commands, like `julia:Execute File`, but I'm less familiar with how those work.

As a last resort, you can also run it in the terminal using `julia examples/openstreetmap.jl`, just like you would run a python script from the terminal. This will be rather slow however.

## Ipython
For exploring stuff I'd recommend using a julia kernel in an Ipython notebook. I've set one up under examples already.
As I understand, you need to first go into the julia repl, and run

```julia
using IJulia
notebook()
```

and follow the instructions to get jupyter set up. I'm not too sure, as it automatically opens for me; I set it up ages ago.

Once that's done, you should be able to open any .ipynb in jupyter, and select julia as the language. Note that this doesn't work from within vscode, but jupyter notebook, jupyter lab, nbviewer, ect all seem to support it.