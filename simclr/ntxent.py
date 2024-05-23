import torch
import torch.nn.functional as F

def ntxent_loss(z_i, z_j, cfg):
    """
    NTXent Loss function.
    Parameters
    ----------
    z_i : torch.tensor
        embedding of original samples (batch_size x emb_size)
    z_j : torch.tensor
        embedding of augmented samples (batch_size x emb_size)
    Returns
    -------
    loss
    """
    tau = cfg['tau']
    z = torch.stack((z_i,z_j), dim=1).view(2*z_i.shape[0], z_i.shape[1])
    a = torch.matmul(z, z.T)
    
    Ls = []
    for i in range(z.shape[0]):
        nn_self = torch.cat([a[i,:i], a[i,i+1:]])
        softmax = F.log_softmax(nn_self/tau, dim=0, dtype=torch.float32)
        Ls.append(softmax[i if i%2 == 0 else i-1])
    Ls = torch.stack(Ls)
    
    loss = torch.sum(Ls) / -z.shape[0]
    return loss

# def ntxent_loss(z_i, z_j, cfg):
#     tau = cfg['tau']
#     batch_size = z_i.shape[0]
#     z = torch.cat([z_i, z_j], dim=0)

#     sim = torch.matmul(z, z.T) / tau
#     sim_ij = torch.diag(sim, batch_size)
#     sim_ji = torch.diag(sim, -batch_size)
#     positives = torch.cat([sim_ij, sim_ji], dim=0)

#     neg_i = torch.cat([sim[:batch_size, :batch_size], sim[:batch_size, batch_size:]], dim=1)
#     neg_j = torch.cat([sim[batch_size:, :batch_size], sim[batch_size:, batch_size:]], dim=1)
#     negatives = torch.cat([neg_i, neg_j], dim=0)

#     logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
#     labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)

#     loss = F.cross_entropy(logits, labels)
#     return loss


# def ntxent_loss(z_i, z_j, cfg):
#     tau = cfg['tau']
#     batch_size = z_i.shape[0]
#     z = torch.cat([z_i, z_j], dim=0)

#     sim = torch.matmul(z, z.T) / tau

#     mask = torch.eye(2 * batch_size, device=z.device).bool()
#     positives = sim[mask].view(2 * batch_size, 1)

#     negatives = sim[~mask].view(2 * batch_size, -1)

#     logits = torch.cat([positives, negatives], dim=1)
#     labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)

#     loss = F.cross_entropy(logits, labels)
#     return loss