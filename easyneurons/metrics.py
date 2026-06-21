import numpy as np
from numpy import dtype
from .general import Tracker

class Metrics(Tracker):
    def __init__(self, every: int = 1, print: bool = True):
        super().__init__(every=every, print=print)

    def add(self, predictions: np.ndarray, answers: np.ndarray) -> None:
        pass

    def clear(self) -> None:
        pass

    def calculate(self, force: bool=True) -> dict | None:
        return None

class CategoricalClassificationMetrics(Metrics):
    def __init__(self, accuracy: bool = True,
                 precision: bool = True,
                 recall: bool = True,
                 f1: bool = True,
                 mcc: bool = False,
                 specificity: bool = False,
                 npv: bool = False,
                 k: bool = False,
                 lr: bool = False,
                 top_k: int = None,
                 every: int = 1,
                 print: bool = True):

        super().__init__(every=every, print=print)

        self.n_classes = None
        self.confusion_matrix = None

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.mcc = mcc
        self.specificity = specificity
        self.npv = npv
        self.k = k
        self.lr = lr
        self.top_k = top_k

        self._top_k_count = 0

    def add(self, predictions: np.ndarray, answers: np.ndarray) -> None:
        if len(answers.shape) == 2:
            answers = np.argmax(answers, axis=1)

        if len(predictions.shape) == 2:
            if self.n_classes is None:
                self.n_classes = len(predictions[0])
                self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

            if self.top_k is not None:
                if self.top_k > self.n_classes:
                    raise ValueError(f"top_k={self.top_k} is greater than number of classes! ({self.n_classes})")

                top_k_indices = np.argpartition(predictions, -self.top_k, axis=1)[:, -self.top_k:]
                self._top_k_count += (top_k_indices == answers[:, None]).any(axis=1).sum()

            predictions = np.argmax(predictions, axis=1)

        else:
            raise ValueError("Wrong shape of an array!")

        self.confusion_matrix += np.bincount(self.n_classes * answers + predictions,
                                             minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)

    def clear(self) -> None:
        self.confusion_matrix[:] = 0
        self._top_k_count = 0

    def calculate(self, force: bool = True) -> dict | None:
        if not force:
            self.calls += 1

            if self.calls % self.every != 0:
                return None

        if self.confusion_matrix is None or self.confusion_matrix.sum() == 0:
            raise ValueError("Confusion Matrix is empty!")

        metrics = { }

        correct_answers = np.diag(self.confusion_matrix)
        confusion_matrix_sum = self.confusion_matrix.sum()

        confusion_matrix_column_sums = self.confusion_matrix.sum(axis=0)
        confusion_matrix_row_sums = self.confusion_matrix.sum(axis=1)

        def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(numerator, denominator,
                             out=np.zeros(self.n_classes),
                             where=denominator != 0)

        # Co faktycznie trzeba policzyć
        need_accuracy = self.accuracy or self.k          # k potrzebuje tylko skalaru accuracy
        need_precision = self.precision or self.f1
        need_recall = self.recall or self.f1 or self.lr
        need_specificity = self.specificity or self.lr
        need_tn = self.accuracy or self.specificity or self.npv or self.lr

        if need_tn:
            FP = confusion_matrix_column_sums - correct_answers
            FN = confusion_matrix_row_sums - correct_answers
            TN = confusion_matrix_sum - correct_answers - FP - FN

        if need_accuracy:
            acc = correct_answers.sum() / confusion_matrix_sum

            if self.accuracy:
                metrics["accuracy"] = acc
                metrics["accuracy_per_class"] = (correct_answers + TN) / confusion_matrix_sum

        if need_precision:
            prec = safe_divide(correct_answers, confusion_matrix_column_sums)

            if self.precision:
                metrics["precision_per_class"] = prec
                metrics["precision"] = np.mean(prec)

        if need_recall:
            rec = safe_divide(correct_answers, confusion_matrix_row_sums)

            if self.recall:
                metrics["recall_per_class"] = rec
                metrics["recall"] = np.mean(rec)

        if self.f1:
            f1_divisor = prec + rec
            f1 = safe_divide(2 * prec * rec, f1_divisor)

            metrics["f1_per_class"] = f1
            metrics["f1"] = np.mean(f1)

        if self.mcc:
            # Współczynnik Matthewsa dla wielu klas (wzór Gorodkina).
            # c = poprawne, s = wszystkie, p_k = sumy kolumn (predykcje), t_k = sumy wierszy (prawda)
            c = correct_answers.sum()
            s = confusion_matrix_sum
            p_k = confusion_matrix_column_sums
            t_k = confusion_matrix_row_sums

            numerator = c * s - np.dot(p_k, t_k)
            denom_left = s ** 2 - np.dot(p_k, p_k)
            denom_right = s ** 2 - np.dot(t_k, t_k)
            denom = np.sqrt(max(0.0, denom_left * denom_right))

            metrics["mcc"] = numerator / denom if denom != 0 else 0.0

        if need_specificity:
            spec = safe_divide(TN, TN + FP)

            if self.specificity:
                metrics["specificity_per_class"] = spec
                metrics["specificity"] = np.mean(spec)

        if self.npv:
            npv = safe_divide(TN, TN + FN)

            metrics["npv_per_class"] = npv
            metrics["npv"] = np.mean(npv)

        if self.k:
            pe = np.dot(confusion_matrix_row_sums, confusion_matrix_column_sums) / (confusion_matrix_sum ** 2)
            denom = 1 - pe

            metrics["k"] = (acc - pe) / denom if denom != 0 else 0.0

        if self.lr:
            # LR+ = Recall / (1 - Specificity)
            # LR- = (1 - Recall) / Specificity

            lr_plus_denom = 1 - spec
            lr_plus = np.divide(rec, lr_plus_denom,
                                 out=np.full(self.n_classes, np.inf),
                                 where=lr_plus_denom != 0)

            lr_minus = np.divide(1 - rec, spec,
                                 out=np.full(self.n_classes, np.inf),
                                 where=spec != 0)

            metrics["lr_plus_per_class"] = lr_plus
            metrics["lr_plus"] = np.mean(lr_plus)

            metrics["lr_minus_per_class"] = lr_minus
            metrics["lr_minus"] = np.mean(lr_minus)

        if self.top_k is not None:
            metrics["top_k"] = self._top_k_count / confusion_matrix_sum

        for key, value in metrics.items():
            self.data.setdefault(key, []).append(value)

        if self.print:
            print(self)

        return metrics

class MultiLabelClassificationMetrics(Metrics):
    def __init__(self, accuracy: bool = True,
                 precision: bool = True,
                 recall: bool = True,
                 f1: bool = True,
                 mcc: bool = False,
                 specificity: bool = False,
                 npv: bool = False,
                 k: bool = False,
                 lr: bool = False,
                 hamming: bool = False,
                 exact: bool = False,
                 threshold: float = 0.5,
                 every: int = 1,
                 print: bool = True):

        super().__init__(every=every, print=print)

        # confusion_matrices[i] = [[TN, FP], [FN, TP]] dla każdej etykiety i, NIE jedna macierz NxN
        self.confusion_matrices = None
        self.n_labels = None

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.mcc = mcc
        self.specificity = specificity
        self.npv = npv
        self.k = k
        self.lr = lr
        self.hamming = hamming
        self.exact = exact

        if threshold > 1 or threshold < 0:
            raise ValueError("!Threshold cannot be higher than 1 or lower than 0!")

        self.threshold = threshold

        self._exact_correct = 0

    def add(self, predictions: np.ndarray, answers: np.ndarray) -> None:
        if self.n_labels is None:
            self.n_labels = predictions.shape[1]
            self.confusion_matrices = np.zeros((self.n_labels, 2, 2))

        prediction_labels = (predictions >= self.threshold).astype(int)
        answers = np.asarray(answers, dtype=int)

        if self.exact:
            matches = np.all(prediction_labels == answers, axis=1)
            self._exact_correct += matches.sum()

        # Zwektoryzowane zliczanie 2x2 per etykieta (bez pętli po etykietach).
        # confusion_matrices[i] = [[TN, FP], [FN, TP]]
        self.confusion_matrices[:, 0, 0] += (~ans & ~pred).sum(axis=0)  # TN
        self.confusion_matrices[:, 0, 1] += (~ans & pred).sum(axis=0)  # FP
        self.confusion_matrices[:, 1, 0] += (ans & ~pred).sum(axis=0)  # FN
        self.confusion_matrices[:, 1, 1] += (ans & pred).sum(axis=0)  # TP

    def clear(self) -> None:
        self.confusion_matrices[:] = 0
        self._exact_correct = 0

    def calculate(self, force: bool = True) -> dict | None:
        if not force:
            self.calls += 1

            if self.calls % self.every != 0:
                return None

        if self.confusion_matrices is None or self.confusion_matrices.sum() == 0:
            raise ValueError("Confusion Matrices are empty!")

        metrics = {}

        TN, FP, FN, TP = (self.confusion_matrices[:, 0, 0], self.confusion_matrices[:, 0, 1],
                          self.confusion_matrices[:, 1, 0], self.confusion_matrices[:, 1, 1])

        def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(numerator, denominator,
                             out=np.zeros_like(numerator, dtype=float),
                             where=denominator != 0)

        # Wartości pomocnicze liczone tylko gdy są potrzebne
        need_accuracy = self.accuracy or self.k
        need_precision = self.precision or self.f1
        need_recall = self.recall or self.f1 or self.lr
        need_specificity = self.specificity or self.lr

        acc_per_label = safe_divide(TP + TN, TP + TN + FP + FN) if need_accuracy else None
        prec = safe_divide(TP, TP + FP) if need_precision else None
        rec = safe_divide(TP, TP + FN) if need_recall else None
        spec = safe_divide(TN, TN + FP) if need_specificity else None

        if self.accuracy:
            metrics["accuracy_per_label"] = acc_per_label
            metrics["accuracy"] = acc_per_label.mean()

        if self.precision:
            metrics["precision_per_label"] = prec
            metrics["precision"] = prec.mean()

        if self.recall:
            metrics["recall_per_label"] = rec
            metrics["recall"] = rec.mean()

        if self.f1:
            denum = prec + rec
            f1 = safe_divide(2 * prec * rec, denum)

            metrics["f1_per_label"] = f1
            metrics["f1"] = f1.mean()

        if self.specificity:
            metrics["specificity_per_label"] = spec
            metrics["specificity"] = spec.mean()

        if self.npv:
            npv = safe_divide(TN, TN + FN)

            metrics["npv_per_label"] = npv
            metrics["npv"] = npv.mean()

        if self.mcc:
            mcc_denum = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            mcc = safe_divide(TP * TN - FP * FN, mcc_denum)

            metrics["mcc_per_label"] = mcc
            metrics["mcc"] = mcc.mean()

        if self.k:
            p_e = safe_divide(((TP + FP) * (TP + FN)) + ((TN + FN) * (TN + FP)),
                              (TN + FP + FN + TP) ** 2)
            kappa = safe_divide(acc_per_label - p_e, 1 - p_e)

            metrics["kappa_per_label"] = kappa
            metrics["kappa"] = kappa.mean()

        if self.lr:
            # LR+ = Recall / (1 - Specificity)
            # LR- = (1 - Recall) / Specificity
            lr_plus_denom = 1 - spec
            lr_plus = np.divide(rec, lr_plus_denom,
                             out=np.full_like(rec, np.inf, dtype=float),
                             where=lr_plus_denom != 0)

            lr_minus_numerator = 1 - rec
            lr_minus = np.divide(lr_minus_numerator, spec,
                             out=np.full_like(lr_minus_numerator, np.inf, dtype=float),
                             where=spec != 0)

            metrics["lr_plus_per_label"] = lr_plus
            metrics["lr_plus"] = lr_plus.mean()

            metrics["lr_minus_per_label"] = lr_minus
            metrics["lr_minus"] = lr_minus.mean()

        if self.hamming:
            # Hamming loss = odsetek błędnych etykiet = (FP + FN) / total
            ham_per_label = safe_divide(FP + FN, TN + TP + FN + FP)

            metrics["hamming_loss_per_label"] = ham_per_label
            metrics["hamming_loss"] = ham_per_label.mean()

        if self.exact:
            metrics["exact_match"] = self._exact_correct / self.confusion_matrices[0].sum()

        for key, value in metrics.items():
            self.data.setdefault(key, []).append(value)

        if self.print:
            print(self)

        return metrics
