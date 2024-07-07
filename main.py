import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RangeSlider
from sklearn.datasets import make_regression
# plt.style.use('dark_background')

N = 1
M = 50

X, y = make_regression(n_samples=M, n_features=N, noise=10, random_state=False)
y = np.expand_dims(y, axis=1)

L = np.array((np.min(X), np.max(X), np.min(y), np.max(y)))

min_learning_rate, init_learning_rate, max_learning_rate = 0.1, 0.5, 2.0
min_epochs, init_epochs_from, init_epochs_to, max_epochs = 0, 1, 12, 15

cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()['color']


# def f(w, b):
#     return [[cost(X * w[i,j] + b[i,j], y) for j in range(M)] for i in range(M)]


def f(w, b):
    return np.mean([0.5 * (x * w + b - y) ** 2 for x, y in zip(X, y)], axis=0)


def cost(a, y):
    return 0.5 * np.mean((a - y) ** 2)


def cost_derivative(a, y):
    return a - y


def gradient(dj_da, x):
    dj_dw = (1 / M) * x.T @ dj_da
    dj_db = (1 / M) * np.sum(dj_da, axis=0, keepdims=True)
    
    return dj_dw, dj_db


def params(learning_rate, epochs_from, epochs_to):
    w = np.full((N, 1), 1e-6)
    b = np.zeros((1, 1))
    p = np.empty((0, 3))

    for epoch in range(epochs_to+1):
        a = X @ w + b
        p = np.vstack((p, [w.squeeze(), b.squeeze(), cost(a, y)]))
            
        dj_dw, dj_db = gradient(cost_derivative(a, y), X)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

    return p[epochs_from:].T


def lines_data(X_min, X_max, y_min, y_max, w, b):
    xs1 = (y_min - b) / w
    xs2 = (y_max - b) / w
    
    xs = np.array((np.minimum(np.maximum(X_min, xs1), np.maximum(X_min, xs2)), 
                   np.maximum(np.minimum(X_max, xs2), np.minimum(X_max, xs1))))
    
    return np.dstack((xs.T, (xs*w+b).T))


def limits(X_min, X_max, y_min, y_max):
    xm = 0.05 * (X_max - X_min)
    ym = 0.05 * (y_max - y_min)

    return dict(xlim=(X_min - xm, X_max + xm), ylim=(y_min - ym, y_max + ym))


def limits1():
    return limits(*L)


def limits2(w, b, j):
    i = np.argpartition(j, 1)[:2]

    wt = np.median(w[i])
    bt = np.median(b[i])

    r = np.max((np.max(w) - wt, wt - np.min(w), np.max(b) - bt, bt - np.min(b)))

    return limits(wt - r, wt + r, bt - r, bt + r)


def contours(xlim, ylim):
    w, b = np.meshgrid(np.linspace(*xlim), np.linspace(*ylim))  
    cs = ax2.contour(w, b, f(w, b), levels=19, linewidths=0.5)
    cl = ax2.clabel(cs, inline=True, fontsize=10)
    
    return cs, cl


def setup1(w, b, j, c):
    ax1.set(**limits1())

    values = ax1.scatter(X, y)
    update1(w, b, j, c)

    return values


def setup2(w, b, j, c):
    lim = limits2(w, b, j)
    
    ax2.set(**lim) 
    cs, cl = contours(**lim)

    trail, = ax2.plot(w, b, ':', linewidth=0.5)  
    trail_ = ax2.scatter(w, b, zorder=2, marker=(4, 1))

    trail_.set_color(c)
    trail_.set_sizes((np.eye(j.size)[np.argmin(j)]*2+1)*36.0)

    return [cs, cl], trail, trail_


def update1(w, b, j, c):
    ax1.set_prop_cycle(plt.cycler(c=c))

    for obj in ax1.lines:
        obj.remove()

    lines = ax1.plot(*np.c_[lines_data(*L, w, b)].T, ':', marker=(4, 1), linewidth=0.5)
    lines[np.argmin(j)].set(markersize=10.0, linestyle='-')
    
    ax1.legend(lines, j.round(1), loc=2)


def update2(w, b, j, c):
    trail.set_data(w, b)
    trail_.set_offsets(np.c_[w, b])

    trail_.set_color(c)
    trail_.set_sizes((np.eye(j.size)[np.argmin(j)]*2.+1.)*36.)

_val = None # there is a bug in the RangeSlider
def update(val):
    global _val
    if _val == val and val:
        return
    _val = val

    (w, b, j), c = params(lr_sl.val, *ep_sl.val), np.roll(cycle_colors, -ep_sl.val[0])

    update1(w, b, j, c)
    update2(w, b, j, c)


def relim(event):
    for obj in q[0].collections:
        obj.remove()

    for obj in q[1]:
        obj.remove()

    ax1.set(**limits1())
    lim = limits2(*params(lr_sl.val, *ep_sl.val))  
    
    ax2.set(**lim)
    q[0], q[1] = contours(**lim)

    fig.canvas.draw_idle()


def reset(event):
    lr_sl.reset()
    ep_sl.reset()


def rand(event):
    X_, y_ = make_regression(n_samples=M, n_features=N, noise=10)
    
    X[:, 0] = X_.squeeze()
    y[:, 0] = y_

    L[:] = np.min(X), np.max(X), np.min(y), np.max(y)

    values.set_offsets(np.c_[X, y])

    if lr_sl.val == init_learning_rate and ep_sl.val == (init_epochs_from, init_epochs_to):
        update(0)
    else:
        reset(0)

    relim(0)


def setup():
    lr_sl = Slider(ax3, 'α', min_learning_rate, max_learning_rate, valinit=init_learning_rate, valstep=0.001)
    lr_sl.on_changed(update)

    ep_sl = RangeSlider(ax4, 'epochs', min_epochs, max_epochs, valinit=(init_epochs_from, init_epochs_to), valstep=1)
    ep_sl.on_changed(update)

    relim_bt = Button(ax5, 'Relim')
    relim_bt.on_clicked(relim)

    reset_bt = Button(ax6, 'Reset')
    reset_bt.on_clicked(reset)

    rand_bt = Button(ax7, 'Rand')
    rand_bt.on_clicked(rand)

    (w, b, j), c = params(init_learning_rate, init_epochs_from, init_epochs_to), np.roll(cycle_colors, -init_epochs_from)
    
    return setup1(w, b, j, c), setup2(w, b, j, c), lr_sl, ep_sl, rand_bt, reset_bt, relim_bt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.7))

ax3 = fig.add_axes([0.124, 0.15, 0.777, 0.03])
ax4 = fig.add_axes([0.124, 0.1, 0.777, 0.03])
ax5 = fig.add_axes([0.6, 0.02, 0.09, 0.04])
ax6 = fig.add_axes([0.72, 0.02, 0.09, 0.04])
ax7 = fig.add_axes([0.84, 0.02, 0.09, 0.04])

values, (q, trail, trail_), lr_sl, ep_sl, rand_bt, reset_bt, relim_bt = setup()

plt.subplots_adjust(bottom=0.25)
plt.show()

