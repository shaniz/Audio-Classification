import os

from tqdm import tqdm

import utils
import validate


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, split,
                       scheduler=None, model_num="", layer=""):
    best_acc = 0.0

    for epoch in range(params.epochs):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)

        acc = validate.evaluate(model, device, val_loader)
        print("Epoch {}/{} Loss:{} Valid Acc:{}".format(epoch, params.epochs, avg_loss, acc))

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
            best_acc_epoch = epoch
        if scheduler:
            scheduler.step()

        # Another subdirectory for deepensemble training
        # model_num- for C
        # layer - for B3
        checkpoint_dir = os.path.join(params.checkpoint_dir, str(model_num), str(layer))
        if is_best or (epoch > 0 and epoch % 20 == 0) or epoch == 69 or epoch == 449:  # 70 or 450
            utils.save_checkpoint({"epoch": epoch + 1,
                                   "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(checkpoint_dir))
        writer.add_scalar("data{}/trainingLoss{}".format(params.dataset_name, split), avg_loss, epoch)
        writer.add_scalar("data{}/valAcc{}".format(params.dataset_name, split), acc, epoch)
    writer.close()
    return acc, best_acc, best_acc_epoch


def lr_lambda(epoch):
    if epoch == 299 | epoch == 349:  # 300 or 350
        return 0.1
    return 1.0
