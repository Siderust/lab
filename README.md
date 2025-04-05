# siderust-py
siderust-py is a modern Python library designed to calculate and track the trajectories and elevations of celestial objects. Whether you’re building an astronomical application, planning a satellite’s path, or simply exploring orbital mechanics, siderust provides easy-to-use interfaces and robust functionality for accurate results.

# Using Docker

### Building the Docker image

   ```bash
   docker build -t siderust-lab .
   ```

### Running the image

   ```bash
   docker run --rm -it -v $PWD:/home/user/src/siderust-lab
   docker run \
      --rm -it \
      -v $PWD:/home/user/src/ \
      -w /home/user/src/ \
      siderust-lab
   ```

### **Build the Python Module**
Build the Rust code into a Python module using `maturin` or `setuptools-rust`.

#### Using `maturin`:
1. Install maturin:
   ```bash
   pip install maturin
   ```
2. Build and install the Python module:
   ```bash
   maturin develop --release
   ```
