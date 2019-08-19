
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy


class Highlighter(object):

    def __init__(self):
        self.color_map = [u'#ff0000', u'#ff4000', u'#ff8000', u'#ffbf00', u'#ffff00', u'#bfff00', u'#80ff00', u'#40ff00', u'#00ff00', u'#00ff40', u'#00ff80', u'#00ffbf',
                          u'#00ffff', u'#00bfff', u'#0080ff', u'#0040ff', u'#0000ff', u'#4000ff', u'#8000ff', u'#bf00ff', u'#ff00ff', u'#ff00bf', u'#ff0080', u'#ff0040', u'#ff0000']

    def highlight(self, activation, text):
        activation = activation.detach().cpu().numpy()
        step_size = ((max(activation) - min(activation)) / len(self.color_map))
        lookup = numpy.array(
            list(numpy.arange(min(activation), max(activation), step_size)))
        colors = []
        for (i, act) in enumerate(activation):
            try:
                colors.append(
                    self.color_map[numpy.where((act > lookup))[0][(- 1)]])
            except IndexError:
                colors.append((len(self.color_map) - 1))
        str_ = u'<br><br>'
        for (i, (char, color)) in enumerate(zip(list(text), colors)):
            str_ += self._render(char, color)
            if (((i % 100) == 0) and (i > 0)):
                str_ += u'<br>'
        return str_

    def highlight_selection(self, activations, text, file_=u'resources/data/highlight.html', n=10):
        ix = numpy.random.choice(activations.shape[1], size=n)
        rendered = u''
        for i in ix:
            rendered += self.highlight(activations[:, i], text)
        with open(file_, u'w') as f:
            f.write(rendered)

    def _render(self, char, color):
        return u'<span style="background-color: {}">{}</span>'.format(color, char)
