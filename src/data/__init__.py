"""
STGAT Project Data Modules

Production-ready data handling utilities for both notebooks and GCP deployment.
"""

from .evaluation_cases import EvaluationCaseManager, load_evaluation_cases, get_case_train_test_data

__all__ = ['EvaluationCaseManager', 'load_evaluation_cases', 'get_case_train_test_data']
