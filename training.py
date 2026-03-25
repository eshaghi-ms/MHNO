import torch
import torch.nn.functional as F
from timeit import default_timer


def train_fno(model, myloss, epochs, batch_size, train_loader, test_loader,
              optimizer, scheduler, normalized, normalizer, device):
    train_mse_log = []
    train_l2_log = []
    test_l2_log = []

    if normalized:
        # a_normalizer = normalizer[0].to(device)
        y_normalizer = normalizer[1].to(device)
    else:
        # a_normalizer = None
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        total_samples = 0
        train_mse_sum = 0
        train_l2_sum = 0
        for x, y in train_loader:
            bsz = x.shape[0]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse = F.mse_loss(out.view(bsz, -1), y.view(bsz, -1), reduction='mean')
            loss = myloss(out.view(bsz, -1), y.view(bsz, -1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_mse_sum += mse.item()
            train_l2_sum += loss.item()
            total_samples += bsz

        train_mse_avg = train_mse_sum / len(train_loader)
        train_l2_avg = train_l2_sum / total_samples

        model.eval()
        total_samples = 0
        test_l2_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                bsz = x.shape[0]
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_l2_sum += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
                total_samples += bsz

        test_l2_avg = test_l2_sum / total_samples

        train_mse_log.append(train_mse_avg)
        train_l2_log.append(train_l2_avg)
        test_l2_log.append(test_l2_avg)

        t2 = default_timer()
        print(ep, t2 - t1, train_mse_avg, train_l2_avg, test_l2_avg)

    return model, train_mse_log, train_l2_log, test_l2_log


def train_fno_time(model, myloss, epochs, batch_size, train_loader, test_loader,
                   optimizer, scheduler, normalized, normalizer, device):
    # train_mse_log = []
    train_l2_log = []
    test_l2_log = []
    step = 1
    # if normalized:
    #     a_normalizer = normalizer[0].to(device)
    #     y_normalizer = normalizer[1].to(device)
    # else:
    #     a_normalizer = None
    #     y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        total_train_samples = 0
        train_l2_sum = 0
        for xx, yy in train_loader:
            bsz = xx.shape[0]
            # loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            T = yy.shape[-1]
            for t in range(0, T, step):
                # y = yy[..., t:t + step]
                im = model(xx)
                # loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

            # train_l2_step += loss.item()
            loss = myloss(pred.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1))
            train_l2_sum += loss.item()
            total_train_samples += bsz

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_l2 = train_l2_sum / total_train_samples
        train_l2_log.append(train_l2)

        model.eval()
        total_test_samples = 0
        test_l2_sum = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                bsz = xx.shape[0]
                # loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                T = yy.shape[-1]
                for t in range(0, T, step):
                    # y = yy[..., t:t + step]
                    im = model(xx)
                    # loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                # test_l2_step += loss.item()
                loss_test = myloss(pred.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1))
                test_l2_sum += loss_test.item()
                total_test_samples += bsz

        test_l2 = test_l2_sum / total_test_samples
        test_l2_log.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)

    return model, train_l2_log, test_l2_log
