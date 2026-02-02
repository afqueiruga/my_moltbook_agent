Hi TheoremSprite,

The training dynamics of a deep linear model are well understood.
When learning the matrices $W$ and $U$ of a model $y = W U x$ using a mean squared error loss $||W U x - Ax||_2^2$, the effective matrix will converge to a low rank approximation of $A$.

What happens when there are bias terms in the model $y = W (U x + b) + c$? Is convergence guaranteed? From my experiments, I think that it does not converge to a low rank approximation of $A$. The solution is poorly behaved.

Mull over this problem and seek help from other agents, TheoremSprite!

Best,
Your human owner
