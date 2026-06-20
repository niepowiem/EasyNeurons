import numpy as np
from numpy import dtype
from .general import Tracker

class Metrics(Tracker):
    def __init__(self, every: int = 1):
        super().__init__(every=every)

    def add(self, predictions: np.ndarray, answers: np.ndarray):
        pass

    def reset(self):
        pass

    def compute(self) -> dict:
        return None

class ClassificationMetrics(Metrics):
    def __init__(self, accuracy:bool = True,
                 precision:bool = False,
                 recall:bool = False,
                 f1:bool = False,
                 specificity:bool = False,
                 every: int = 1):

        super().__init__(every=every)

        self.n_classes = 1
        self.confusion_matrix = np.array([[0]])

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.specificity = specificity

    def add(self, predictions: np.ndarray, answers: np.ndarray):
        if len(answers.shape) == 2:
            answers = np.argmax(answers, axis=1)

        if len(predictions.shape) == 2:
            predictions = np.argmax(predictions, axis=1)

        current_c_classes = max(answers.max(), predictions.max()) + 1
        if current_c_classes > self.n_classes:
            difference = current_c_classes - self.n_classes

            self.confusion_matrix = np.pad(self.confusion_matrix, ((0, difference), (0, difference)))
            self.n_classes = current_c_classes

        self.confusion_matrix += np.bincount(self.n_classes * answers + predictions, minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)

    def reset(self):
        self.confusion_matrix[:] = 0

    @property
    def npv(self) -> np.ndarray:
        # Negative Predictive Value
        # TN / (TN + FN)
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP

        FN = self.confusion_matrix.sum(axis=1) - TP
        TN = self.confusion_matrix.sum() - TP - FP - FN

        divisor = TN + FN

        return np.divide(TN, divisor,
                         out=np.zeros(self.n_classes),
                         where=divisor != 0)

    @property
    def lr_plus(self) -> np.ndarray:
        # Positive Likelihood Ratio (LR+)
        # Sensitivity / (1 - Specificity)  =>  TPR / FPR
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP

        FN = self.confusion_matrix.sum(axis=1) - TP
        TN = self.confusion_matrix.sum() - TP - FP - FN

        tpr_divisor = TP + FN
        tpr = np.divide(TP, tpr_divisor, out=np.zeros(self.n_classes), where=tpr_divisor != 0)

        fpr_divisor = FP + TN
        fpr = np.divide(FP, fpr_divisor, out=np.zeros(self.n_classes), where=fpr_divisor != 0)

        return np.divide(tpr, fpr,
                         out=np.zeros(self.n_classes),
                         where=fpr != 0)

    @property
    def lr_minus(self) -> np.ndarray:
        # Negative Likelihood Ratio (LR-)
        # (1 - Sensitivity) / Specificity  =>  FNR / TNR
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP

        FN = self.confusion_matrix.sum(axis=1) - TP
        TN = self.confusion_matrix.sum() - TP - FP - FN

        fnr_divisor = TP + FN
        fnr = np.divide(FN, fnr_divisor, out=np.zeros(self.n_classes), where=fnr_divisor != 0)

        tnr_divisor = TN + FP
        tnr = np.divide(TN, tnr_divisor, out=np.zeros(self.n_classes), where=tnr_divisor != 0)

        return np.divide(fnr, tnr,
                         out=np.zeros(self.n_classes),
                         where=tnr != 0)

    @property
    def mcc(self) -> np.ndarray:
        # Matthews Correlation Coefficient
        # (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        # Rzutowanie na float64 zapobiega przepełnieniu (integer overflow) przy mnożeniu w mianowniku
        TP = np.diag(self.confusion_matrix).astype(np.float64)
        FP = (self.confusion_matrix.sum(axis=0) - TP).astype(np.float64)

        FN = (self.confusion_matrix.sum(axis=1) - TP).astype(np.float64)
        TN = (self.confusion_matrix.sum() - TP - FP - FN).astype(np.float64)

        numerator = (TP * TN) - (FP * FN)
        denominator_squared = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        denominator = np.sqrt(denominator_squared)

        return np.divide(numerator, denominator,
                         out=np.zeros(self.n_classes),
                         where=denominator != 0)

    @property
    def cohens_kappa(self) -> float:
        # (p_o - p_e) / (1 - p_e)
        # Gdzie p_o to obserwowana zgodność (accuracy), a p_e to oczekiwana zgodność
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0

        # p_o to po prostu dokładność (accuracy)
        p_o = np.diag(self.confusion_matrix).sum() / total

        # p_e: suma iloczynów sum wierszy i kolumn, podzielona przez kwadrat całkowitej sumy
        sum_rows = self.confusion_matrix.sum(axis=1)
        sum_cols = self.confusion_matrix.sum(axis=0)
        p_e = np.sum(sum_rows * sum_cols) / (total ** 2)

        divisor = 1.0 - p_e

        if divisor == 0:
            return 0.0

        return float((p_o - p_e) / divisor)
    
    
    def compute(self, force: bool=True) -> dict:
        if not force:
            if self.calls % self.every != 0:
                self.calls += 1

                return

        metrics = { }

        correct_answers = np.diag(self.confusion_matrix)
        confusion_matrix_sum = self.confusion_matrix.sum()

        confusion_matrix_column_sums = self.confusion_matrix.sum(axis=0) if self.precision or self.f1 or self.specificity else None
        confusion_matrix_row_sums = self.confusion_matrix.sum(axis=1) if self.precision or self.f1 or self.specificity else None

        if self.accuracy:
            acc = correct_answers / confusion_matrix_sum

            metrics["accuracy"] = acc.sum()
            metrics["accuracy_per_class"] = acc * self.n_classes

        if self.precision or self.f1:
            prec = np.divide(correct_answers, confusion_matrix_column_sums,
                                             out=np.zeros(self.n_classes),
                                             where=confusion_matrix_column_sums != 0)

            metrics["precision"] = prec.mean()
            metrics["precision_per_class"] = prec

        if self.recall or self.f1:
            rec = np.divide(correct_answers, confusion_matrix_row_sums,
                                          out=np.zeros(self.n_classes),
                                          where=confusion_matrix_row_sums != 0)

            metrics["recall"] = rec.mean()
            metrics["recall_per_class"] = rec

        if self.f1:
            f1_divisor = metrics["precision"] + metrics["recall"]
            f1_result = np.divide(2 * metrics["recall"] * metrics["precision"], f1_divisor,
                                      out=np.zeros(self.n_classes),
                                      where=f1_divisor != 0)

            metrics["f1"] = f1_result.mean()
            metrics["f1_per_class"] = f1_result

        if self.specificity:
            FP = confusion_matrix_column_sums - correct_answers
            TN = confusion_matrix_sum - correct_answers - FP - (confusion_matrix_row_sums - correct_answers)
            specificity_divisor = TN + FP

            spec = np.divide(TN, specificity_divisor,
                                               out=np.zeros(self.n_classes),
                                               where=specificity_divisor != 0)

            metrics["specificity"] = spec.mean()
            metrics["specificity_per_class"] = spec

        for key, value in metrics.items():
            self.data.setdefault(key, []).append(value)

        return metrics


