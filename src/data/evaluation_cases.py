"""
Production-ready evaluation case management for STGAT project

This module provides consistent evaluation case handling across
notebooks and GCP deployment environments.
"""

import json
import pandas as pd
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime

class EvaluationCaseManager:
    """
    Manages evaluation cases for consistent model comparison
    """
    def __init__(self, cases_filepath: str = None):
        if cases_filepath is None:
            # Auto-detect path based on current working directory
            if os.path.exists('results/evaluation_cases.json'):
                # Running from project root
                self.cases_filepath = 'results/evaluation_cases.json'
            elif os.path.exists('../results/evaluation_cases.json'):
                # Running from notebooks directory
                self.cases_filepath = '../results/evaluation_cases.json'
            else:
                # Fallback - assume project root structure
                self.cases_filepath = 'results/evaluation_cases.json'
        else:
            self.cases_filepath = cases_filepath
        self.cases_data = self.load_cases()
    
    def load_cases(self) -> Dict[str, Any]:
        """Load evaluation cases from JSON file"""
        try:
            with open(self.cases_filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Cases file not found at {self.cases_filepath}")
            return {'metadata': {}, 'cases': []}
    
    def get_cases_list(self) -> List[Dict[str, Any]]:
        """Get list of evaluation cases"""
        return self.cases_data.get('cases', [])
    
    def get_case_data(self, sales_data: pd.DataFrame, 
                     case_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test data for specific evaluation case
        
        Args:
            sales_data: Complete sales dataset
            case_info: Case information dictionary
            
        Returns:
            Tuple of (train_data, test_data)
        """
        store = case_info['store_nbr']
        family = case_info['family']
        
        case_data = sales_data[
            (sales_data['store_nbr'] == store) & 
            (sales_data['family'] == family)
        ].sort_values('date').copy()
        
        # Use standard test split date
        test_split = pd.to_datetime('2017-07-01')
        train_data = case_data[case_data['date'] < test_split]
        test_data = case_data[case_data['date'] >= test_split]
        
        return train_data, test_data
    
    def validate_cases_coverage(self, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that all evaluation cases have adequate data coverage
        """
        coverage_report = {
            'validation_date': datetime.now().isoformat(),
            'total_cases': len(self.get_cases_list()),
            'valid_cases': 0,
            'case_details': [],
            'coverage_summary': {}
        }
        
        for case in self.get_cases_list():
            train_data, test_data = self.get_case_data(sales_data, case)
            
            case_validation = {
                'case_id': case.get('case_id'),
                'store_nbr': case['store_nbr'],
                'family': case['family'],
                'train_records': len(train_data),
                'test_records': len(test_data),
                'train_date_range': {
                    'start': train_data['date'].min() if len(train_data) > 0 else None,
                    'end': train_data['date'].max() if len(train_data) > 0 else None
                },
                'test_date_range': {
                    'start': test_data['date'].min() if len(test_data) > 0 else None,
                    'end': test_data['date'].max() if len(test_data) > 0 else None
                },
                'avg_train_sales': train_data['sales'].mean() if len(train_data) > 0 else 0,
                'avg_test_sales': test_data['sales'].mean() if len(test_data) > 0 else 0
            }
            
            # Validation criteria
            if (len(train_data) >= 150 and len(test_data) >= 30 and 
                train_data['sales'].mean() >= 5):
                coverage_report['valid_cases'] += 1
                case_validation['validation_status'] = 'valid'
            else:
                case_validation['validation_status'] = 'invalid'
            
            coverage_report['case_details'].append(case_validation)
        
        coverage_report['coverage_summary'] = {
            'validation_rate': coverage_report['valid_cases'] / coverage_report['total_cases'] if coverage_report['total_cases'] > 0 else 0,
            'avg_train_records': sum(c['train_records'] for c in coverage_report['case_details']) / len(coverage_report['case_details']) if coverage_report['case_details'] else 0,
            'avg_test_records': sum(c['test_records'] for c in coverage_report['case_details']) / len(coverage_report['case_details']) if coverage_report['case_details'] else 0
        }
        
        return coverage_report
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluation cases metadata"""
        return self.cases_data.get('metadata', {})

# Convenience functions for direct use
def load_evaluation_cases(filepath: str = None) -> List[Dict[str, Any]]:
    """Load evaluation cases directly"""
    manager = EvaluationCaseManager(filepath)
    return manager.get_cases_list()

def get_case_train_test_data(sales_data: pd.DataFrame, store_nbr: int, 
                           family: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get train/test data for specific store-family combination"""
    case_info = {'store_nbr': store_nbr, 'family': family}
    manager = EvaluationCaseManager()
    return manager.get_case_data(sales_data, case_info)
