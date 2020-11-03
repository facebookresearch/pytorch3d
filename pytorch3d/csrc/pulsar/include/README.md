# The `include` folder

This folder contains header files with implementations of several useful
algorithms. These implementations are usually done in files called `x.device.h`
and use macros that route every device specific command to the right
implementation (see `commands.h`).

If you're using a device specific implementation, include `x.device.h`.
This gives you the high-speed, device specific implementation that lets
you work with all the details of the datastructure. All function calls are
inlined. If you need to work with the high-level interface and be able to
dynamically pick a device, only include `x.h`. The functions there are
templated with a boolean `DEV` flag and are instantiated in device specific
compilation units. You will not be able to use any other functions, but can
use `func<true>(params)` to work on a CUDA device, or `func<false>(params)`
to work on the host.
