"""evaluation metrics for plagiarism detection"""

from typing import List, Dict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


class EvaluationMetrics:
    """computes standard classification metrics"""

    @staticmethod
    def calculate_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
        """returns precision, recall, f1, accuracy and confusion matrix values"""
        y_true_binary = [1 if label else 0 for label in y_true]
        y_pred_binary = [1 if label else 0 for label in y_pred]

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

        cm = confusion_matrix(y_true_binary, y_pred_binary)

        # handle edge case where only one class exists in data
        if cm.size == 1:
            if y_true_binary[0] == 1:
                tp = cm[0, 0] if y_pred_binary[0] == 1 else 0
                fn = cm[0, 0] if y_pred_binary[0] == 0 else 0
                fp, tn = 0, 0
            else:
                tn = cm[0, 0] if y_pred_binary[0] == 0 else 0
                fp = cm[0, 0] if y_pred_binary[0] == 1 else 0
                tp, fn = 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr)
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float], system_name: str = "system"):
        """prints metrics in a readable format"""
        print(f"\n{'='*50}")
        print(f"evaluation metrics for {system_name}")
        print(f"{'='*50}")
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall:    {metrics['recall']:.4f}")
        print(f"f1 score:  {metrics['f1_score']:.4f}")
        print(f"accuracy:  {metrics['accuracy']:.4f}")
        print(f"\nconfusion matrix:")
        print(f"  tp: {metrics['true_positives']:3d}  fn: {metrics['false_negatives']:3d}")
        print(f"  fp: {metrics['false_positives']:3d}  tn: {metrics['true_negatives']:3d}")
        print(f"\nerror rates:")
        print(f"  false positive rate: {metrics['false_positive_rate']:.4f}")
        print(f"  false negative rate: {metrics['false_negative_rate']:.4f}")
        print(f"{'='*50}\n")


class ErrorAnalyzer:
    """categorizes prediction errors for debugging"""

    @staticmethod
    def analyze_errors(
        test_cases: List[Dict],
        predictions: List[bool]
    ) -> Dict[str, List[Dict]]:
        """splits errors into false positives and false negatives"""
        false_positives = []
        false_negatives = []

        for i, (test_case, pred) in enumerate(zip(test_cases, predictions)):
            ground_truth = test_case.get('is_plagiarism', False)

            if pred and not ground_truth:
                false_positives.append({
                    'index': i,
                    'test_case': test_case,
                    'predicted': True,
                    'actual': False
                })
            elif not pred and ground_truth:
                false_negatives.append({
                    'index': i,
                    'test_case': test_case,
                    'predicted': False,
                    'actual': True
                })

        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    @staticmethod
    def print_error_analysis(errors: Dict[str, List[Dict]]):
        """shows what went wrong and why"""
        print("\n" + "="*50)
        print("error analysis")
        print("="*50)

        print(f"\nfalse positives: {len(errors['false_positives'])}")
        for fp in errors['false_positives']:
            print(f"  - test case #{fp['index']}: {fp['test_case'].get('description', 'no description')}")

        print(f"\nfalse negatives: {len(errors['false_negatives'])}")
        for fn in errors['false_negatives']:
            print(f"  - test case #{fn['index']}: {fn['test_case'].get('description', 'no description')}")
            if 'transformation_type' in fn['test_case']:
                print(f"    transformation: {fn['test_case']['transformation_type']}")

        print("="*50 + "\n")
