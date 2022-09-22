import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os, torch


def to_cpu_ndarray(obj):
    return obj.cpu().numpy()


class distribution_plot(object):
    def __init__(self, args):
        super(distribution_plot, self).__init__()
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def tsne_plot(self, model, target_test_loader, filename):
        """
        For visualize the feature distribution of two domains.
        batch_size should be the whole number of samples shown in the plot.
        input: the self.model AND self.loader_dict["ts"]
        """
        data, label = next(iter(target_test_loader))
        data, label = data.to(self.device), label.to(self.device)
        model.eval()
        model = model.to(self.device)
        # model.load_state_dict(torch.load(     ))
        with torch.no_grad():
            tgt_out = model.predict(data)
            tgt_outshape = tgt_out.shape
            print("The data size of this batch:{}".format(tgt_outshape))
            tgt_out = tgt_out.reshape(tgt_outshape[0], -1)
            tgt_out = to_cpu_ndarray(tgt_out)
            label = to_cpu_ndarray(label)
            tsne = TSNE(n_components=2)
            X_embeded = tsne.fit_transform(tgt_out)
            sns.set(style='white', rc={'figure.figsize': (5, 5)})
            palette = sns.color_palette("bright", self.args.num_class)
            dirs = "pictures/" + self.args.dataset
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            sns.scatterplot(X_embeded[:, 0], X_embeded[:, 1], hue=label, style=label, legend=True, palette=palette)
        plt.savefig(dirs + '/' + filename + ".svg")
        plt.clf()

