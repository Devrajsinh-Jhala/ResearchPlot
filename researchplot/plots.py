import matplotlib.pyplot as plt
from .config import PLOT_FORMATS
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def bar(data, xticks=None, xlabel='', ylabel='', format="ieee", title=None, output_path=None, yerr=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.bar(range(len(data)), data, yerr=yerr, linewidth=params["linewidth"])
    if xticks:
        plt.xticks(range(len(data)), xticks, fontsize=params["xtick_font_size"])
    plt.xlabel(xlabel, fontsize=params["xlabel_font_size"])
    plt.ylabel(ylabel, fontsize=params["ylabel_font_size"])
    if title:
        plt.title(title, fontsize=params["title_font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
# Stacked bar plot
def stacked_bar(data, stack_labels, xticks=None, xlabel='', ylabel='', format="ieee", title=None, output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    bottom = None
    for i, stack in enumerate(data):
        plt.bar(range(len(stack)), stack, bottom=bottom, label=stack_labels[i], linewidth=params["linewidth"])
        if bottom is None:
            bottom = stack
        else:
            bottom = [sum(x) for x in zip(bottom, stack)]
    plt.xlabel(xlabel, fontsize=params["xlabel_font_size"])
    plt.ylabel(ylabel, fontsize=params["ylabel_font_size"])
    if xticks:
        plt.xticks(range(len(data[0])), xticks, fontsize=params["xtick_font_size"])
    if title:
        plt.title(title, fontsize=params["title_font_size"])
    plt.legend(fontsize=params["legend_font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def scatter(x, y, xlabel, ylabel, format="ieee", output_path=None, 
            show_correlation=True, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.scatter(x, y, s=params["marker_size"], linewidth=params["linewidth"])
    plt.xlabel(xlabel, fontsize=params["font_size"])
    plt.ylabel(ylabel, fontsize=params["font_size"])
    if show_correlation:
        corr = np.corrcoef(x, y)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.2f}', transform=plt.gca().transAxes, 
                 fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
    
def line(x, y, xlabel, ylabel, format="ieee", output_path=None, 
         show_confidence_interval=False, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(x, y, linewidth=params["linewidth"])
    if show_confidence_interval:
        # Code to calculate and plot confidence interval
        pass
    plt.xlabel(xlabel, fontsize=params["font_size"])
    plt.ylabel(ylabel, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()


def histogram(data, bins=None, xlabel='', ylabel='', format="ieee", 
              output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    if bins is None:
        bins = params["hist_bins"]
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.hist(data, bins=bins, linewidth=params["linewidth"])
    plt.xlabel(xlabel, fontsize=params["font_size"])
    plt.ylabel(ylabel, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def boxplot(data, labels=None, xlabel='', ylabel='', format="ieee", 
            output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.boxplot(data, widths=params["box_width"])
    if labels:
        plt.xticks(range(1, len(labels)+1), labels, fontsize=params["font_size"])
    plt.xlabel(xlabel, fontsize=params["font_size"])
    plt.ylabel(ylabel, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def heatmap(data, xticklabels=None, yticklabels=None, xlabel='', ylabel='', 
            format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    heatmap = plt.imshow(data, cmap=params["heatmap_cmap"], aspect='auto')
    plt.colorbar(heatmap)
    if xticklabels:
        plt.xticks(range(len(xticklabels)), xticklabels, fontsize=params["font_size"])
    if yticklabels:
        plt.yticks(range(len(yticklabels)), yticklabels, fontsize=params["font_size"])
    plt.xlabel(xlabel, fontsize=params["font_size"])
    plt.ylabel(ylabel, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def confusion_matrix(cm, classes, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=params["confusion_matrix_cmap"])
    plt.xlabel('Predicted Labels', fontsize=params["font_size"])
    plt.ylabel('True Labels', fontsize=params["font_size"])
    plt.xticks(range(len(classes)), classes, fontsize=params["font_size"])
    plt.yticks(range(len(classes)), classes, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def accuracy_vs_epoch(epochs, accuracy, val_accuracy=None, format="ieee", 
                      output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(epochs, accuracy, linewidth=params["accuracy_loss_linewidth"], label='Training Accuracy')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, linewidth=params["accuracy_loss_linewidth"], label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=params["font_size"])
    plt.ylabel('Accuracy', fontsize=params["font_size"])
    plt.legend(fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def loss_vs_epoch(epochs, loss, val_loss=None, format="ieee", 
                  output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(epochs, loss, linewidth=params["accuracy_loss_linewidth"], label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, linewidth=params["accuracy_loss_linewidth"], label='Validation Loss')
    plt.xlabel('Epochs', fontsize=params["font_size"])
    plt.ylabel('Loss', fontsize=params["font_size"])
    plt.legend(fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def roc_curve(y_true, y_scores, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(fpr, tpr, lw=params["linewidth"], label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate', fontsize=params["font_size"])
    plt.ylabel('True Positive Rate', fontsize=params["font_size"])
    plt.legend(fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def precision_recall_curve(y_true, y_scores, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(recall, precision, lw=params["linewidth"], label=f'AUC = {pr_auc:.2f}')
    plt.xlabel('Recall', fontsize=params["font_size"])
    plt.ylabel('Precision', fontsize=params["font_size"])
    plt.legend(fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()

def violinplot(data, labels=None, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    sns.violinplot(data=data, linewidth=params["linewidth"])
    if labels:
        plt.xticks(range(len(labels)), labels, fontsize=params["font_size"])
    plt.xlabel('Groups', fontsize=params["font_size"])
    plt.ylabel('Values', fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()

def contour_plot(X, Y, Z, levels=None, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    if levels:
        plt.contourf(X, Y, Z, levels=levels, cmap=params["contour_cmap"])
    else:
        plt.contourf(X, Y, Z, cmap=params["contour_cmap"])
    plt.colorbar()
    plt.xlabel('X-axis', fontsize=params["font_size"])
    plt.ylabel('Y-axis', fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def pie(data, labels, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': params["font_size"]})
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()

def hexbin(x, y, gridsize=20, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.hexbin(x, y, gridsize=gridsize, cmap=params.get("hexbin_cmap", "inferno"))
    plt.xlabel('X-axis', fontsize=params["font_size"])
    plt.ylabel('Y-axis', fontsize=params["font_size"])
    plt.colorbar(label='Counts')
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def pairplot(data, variables, hue=None, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    sns.set(style="ticks", font_scale=params["font_scale"])
    sns.pairplot(data, vars=variables, hue=hue, diag_kind="kde", markers="+")
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def learning_curves(train_sizes, train_scores, test_scores, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, test_scores, label='Test Score')
    plt.xlabel('Training size', fontsize=params["font_size"])
    plt.ylabel('Score', fontsize=params["font_size"])
    plt.legend(fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def time_series(time, data, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.plot(time, data, linewidth=params["linewidth"])
    plt.xlabel('Time', fontsize=params["font_size"])
    plt.ylabel('Data', fontsize=params["font_size"])
    plt.xticks(rotation=45)
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()

def radar_chart(categories, data, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    fig = plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    data = np.concatenate((data, [data[0]]))
    ax.plot(angles, data, linewidth=params["linewidth"])
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()

def dendrogram(linkage_matrix, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    dendrogram(linkage_matrix)
    plt.xlabel('Samples', fontsize=params["font_size"])
    plt.ylabel('Distance', fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()
    
def quiver(X, Y, U, V, format="ieee", output_path=None, **kwargs):
    params = PLOT_FORMATS.get(format, PLOT_FORMATS["ieee"]).copy()
    params.update(kwargs)
    plt.figure(figsize=params["figsize"], dpi=params["dpi"])
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.xlabel('X-axis', fontsize=params["font_size"])
    plt.ylabel('Y-axis', fontsize=params["font_size"])
    if output_path:
        plt.savefig(output_path, dpi=params["dpi"])
    plt.show()