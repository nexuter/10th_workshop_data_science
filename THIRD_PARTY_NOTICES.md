# Third-Party Notices

## src/ros_wildfire

The `src/ros_wildfire` package is vendored from the `ros-based-wildfire-prediction`
project (Rothermel + Huygens/Richards wavelet fire-spread simulation with PSO
calibration), authored by Jaeyeon Kihm and licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Jaeyeon Kihm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Only the `ros_wildfire` package itself was vendored (forward Huygens/PSO
prediction: `config`, `data`, `io`, `physics`, `eval`, `calib`, `viz`). The
rest of the `ros-based-wildfire-prediction` repository (its own dataset,
experiment logs, notebooks, and unrelated `transformer` module) is out of
scope for this project and was not copied.
