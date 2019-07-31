.. image:: https://badge.fury.io/py/conebeam-projector.svg
   :target: https://badge.fury.io/py/conebeam-projector
   :alt: PyPI version

==================
conebeam_projector
==================

This is the `CONRAD <https://github.com/akmaier/CONRAD>`_ cone beam projector ported to pycuda.

Install
-------

.. code:: bash

   pip install conebeam-projector

Or from this repo:

.. code:: bash

    pip install -e .

Usage
-----

.. code:: python

   import pyconrad.autoinit
   import pyconrad.config
   import pycuda.gpuarray as gpuarray

   import conebeam_projector
   from edu.stanford.rsl.conrad.phantom import NumericalSheppLogan3D

   phantom = np.array(NumericalSheppLogan3D(
       *pyconrad.config.get_reco_size()).getNumericalSheppLoganPhantom(), np.float32)
   pyconrad.imshow(phantom, "phantom")
   projector = conebeam_projector.CudaProjector()

   sino = gpuarray.zeros(pyconrad.config.get_sino_shape(), np.float32)

   projector.forward_project_cuda_raybased(phantom, sino, use_maximum_intensity_projection=False)
   pyconrad.imshow(sino, "Sinogram")
   backprojection = projector.backProjectPixelDrivenCuda(sino)
   pyconrad.imshow(backprojection, "backprojection")


Configuration
-------------

Configuration of the projector geometry is done by (py)CONRAD.
The first time you use it CONRAD will suggest you to create a global :code:`Conrad.xml` in your home directory which stores all configuration options.
You can launch :code:`conrad` from bash command line to get a GUI loaded.
You can set the configuration programmatically via

.. code:: python

    import pyconrad.autoinit  # launches JVM
    import pyconrad.config
    this_is_the_configuration_obj = pyconrad.config.get_conf()

This will give you a instance of CONRAD's `edu.stanford.rsl.conrad.utils.Configuration <https://github.com/akmaier/CONRAD/blob/master/src/edu/stanford/rsl/conrad/utils/Configuration.java>`_ class.
