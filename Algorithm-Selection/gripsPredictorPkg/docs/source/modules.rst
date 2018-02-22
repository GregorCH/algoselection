API documentation
=================

Configuration
-------------

.. autosummary::
    predictor.config.Config.__init__
    predictor.config.Config.__str__
    predictor.config.Config._set_predicted_value_evaluation
    predictor.config.Config._set_actual_value_evaluation
    predictor.config.Config._init_actual_data
    predictor.config.Config.set_parameter
    predictor.config.Config.is_parameter_undefined
    predictor.config.Config.load_from_file
    predictor.config.Config.save_parameter
    predictor.config.Config.save_to_file
    predictor.config.Config.reload
    predictor.config.Config.check_required_params


.. automodule:: predictor.config
  :members:
  :private-members:


Logger
------

.. automodule:: predictor.logger
  :members:
  :private-members:

Feature Investigator
--------------------

.. automodule:: predictor.feature_investigator.correlations
  :members:

.. automodule:: predictor.feature_investigator.mds
  :members:

.. automodule:: predictor.feature_investigator.pca
  :members:

.. automodule:: predictor.feature_investigator.rfr
  :members:

Models
------

Random Regressor Forest
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: predictor.models.random_forest_regressor
  :members:

Hydra
^^^^^

.. automodule:: predictor.models.hydra
  :members:


Utilities
^^^^^^^^^

.. automodule:: predictor.models.utilities
  :members:


Performance measurement
-----------------------

.. automodule:: predictor.performance.measurement
  :members:

Plotter
-------

.. automodule:: predictor.performance.visualisation.plotter
  :members:
