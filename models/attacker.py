import torch as ch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm


class Attacker(ch.nn.Module):
    
    def __init__(self, model):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.model = model

    def forward(self, x, target, eps, iterations, step_size,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = nn.CrossEntropyLoss(reduction='none')
        #step_class = L2Step()
        step = L2Step(eps=eps, orig_input=orig_input, step_size=step_size) 

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''      
            output = self.model(inp)
            
            if custom_loss:
                loss, logits, labels = custom_loss(output, target)
                return loss

            return criterion(output, target)

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = step.random_perturb(x)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates
            for _ in iterator:
                x = x.clone().detach().requires_grad_(True)
                losses = calc_loss(inp=step.to_image(x), target=target)
                #assert losses.shape[0] == x.shape[0], \
                #        'Shape of losses must match input!'

                loss = ch.mean(losses)

                if step.use_grad:
                    if (est_grad is None) and mixed_precision:
                        with amp.scale_loss(loss, []) as sl:
                            sl.backward()
                        grad = x.grad.detach()
                        x.grad.zero_()
                    elif (est_grad is None):
                        grad, = ch.autograd.grad(m * loss, [x])
                    else:
                        f = lambda _x, _y: m * calc_loss(inp=step.to_image(_x), target=_y)[0]
                        grad = calc_est_grad(f, x, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    x = step.step(x, grad, 0) #0 è bypass
                    x = step.project(x)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best: 
                ret = x.clone().detach()
                return step.to_image(ret) if return_image else ret

            losses = calc_loss(inp=step.to_image(x), target=target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return step.to_image(best_x) if return_image else best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class L2Step():

    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g, bypass):
        """
        """
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)

        # Attack only positive pairs
        # if (bypass == 1):
        #     for i in range(scaled_g.shape[0]):
        #         if i != 0 or i != scaled_g.shape[0]/2:
        #             scaled_g[i, ...] = ch.zeros(size=(scaled_g.shape[1], scaled_g.shape[2], scaled_g.shape[3]))
            

        # # Attack only negative pairs
        # if (bypass == 2):
        #     scaled_g[0, ...] = ch.zeros(size=(scaled_g.shape[1], scaled_g.shape[2], scaled_g.shape[3]))
        #     scaled_g[int(scaled_g.shape[0]/2), ...] = ch.zeros(size=(scaled_g.shape[1], scaled_g.shape[2], scaled_g.shape[3]))


        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = ch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x


