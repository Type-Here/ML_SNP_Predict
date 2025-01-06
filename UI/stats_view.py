from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json

def display_stats_in_textedit(stats_file:str, text_edit:QTextEdit):
    """
    Display model statistics in a QTextEdit widget.

    Parameters:
        stats_file (str): Path to the JSON file containing model statistics.
        text_edit (QTextEdit): The QTextEdit widget to display the statistics.
    """
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Prepare the statistics to display, excluding AUC and Loss plot data
        stats_text = [
            f"Accuracy: {stats['accuracy']:.3f}",
            "Precision per class: " + ", ".join(f"{p:.3f} " for p in stats['precision_per_class']),
            "Recall per class: " + ", ".join(f"{r:.3f} " for r in stats['recall_per_class']),
            "F1 Score per class: " + ", ".join(f"{f:.3f} " for f in stats['f1_per_class']),
            "Specificity per class: " + ", ".join(f"{s:.3f} " for s in stats['specificity_per_class'])
        ]

        # Set the text in the QTextEdit
        text_edit.setPlainText("\n".join(stats_text))
        text_edit.setReadOnly(True)

    except FileNotFoundError:
        text_edit.setPlainText("Warning: Statistics file not found.")
    except json.JSONDecodeError:
        text_edit.setPlainText("Error: Unable to decode the statistics file.")




def display_plots_in_layout(stats_file, layout:QLayout, text_edit):
    """
    Display AUC and Loss plots in a QVBoxLayout.

    Parameters:
        stats_file (str): Path to the JSON file containing model statistics.
        layout (QVBoxLayout): The Layout in which to display the plots.
        text_edit (QTextEdit): The QTextEdit widget to display warnings or errors.
    """
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Prepare the figures
        figure = Figure()
        canvas = FigureCanvas(figure)

        layout.setContentsMargins(10, 10, 10, 10)  # Set margins
        layout.addWidget(canvas)  # Add canvas to the layout

        # Plot AUC
        ax_auc = figure.add_subplot(211)  # First subplot
        for class_id, data in stats['fpr_tpr'].items():
            ax_auc.plot(data['fpr'], data['tpr'], label=f"Class {class_id} (AUC: {stats['roc_auc'][class_id]:.2f})")
        ax_auc.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        ax_auc.set_title("ROC Curve")
        ax_auc.set_xlabel("False Positive Rate")
        ax_auc.set_ylabel("True Positive Rate")
        ax_auc.legend(loc="lower right", bbox_to_anchor=(1, 0))  # Legend outside the plot

        # Plot Loss
        ax_loss = figure.add_subplot(212)  # Second subplot
        ax_loss.plot(stats['loss_data'], label="Loss")
        ax_loss.plot(stats['val_loss_data'], label="Validation Loss")
        ax_loss.set_title("Loss During Training")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(loc="upper right", bbox_to_anchor=(1, 1))

        # Optimize layout
        figure.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to fit the canvas

        # Refresh the canvas
        canvas.setMaximumWidth(600)  
        canvas.setMinimumWidth(300) 
        canvas.setMinimumHeight(500)
        canvas.draw()

    except FileNotFoundError:
        text_edit.setPlainText("Warning: Statistics file not found.")
    except json.JSONDecodeError:
        text_edit.setPlainText("Error: Unable to decode the statistics file.")




def get_model_name_from_common_name(common_name:str):
    """
    Get the model name from the common name of the model.

    Parameters:
        common_name (str): The common name of the model.

    Returns:
        str: The model name.
    """
    if common_name == "P53 Model":
        return "p53_model"
    elif common_name == "P53 Pfam":
        return "p53_pfam_model"
    elif common_name == "HRAS Transfer":
        return "hras_model"
    else:
        return None