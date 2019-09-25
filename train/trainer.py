import logging
from torch.utils.tensorboard import SummaryWriter
from utils.utils_common import DataModes
import torch
logger = logging.getLogger(__name__)

class Trainer(object):

    def training_step(self, data, epoch):

        # Get the minibatch
        x, y = data


        self.optimizer.zero_grad()
        loss, log = self.net.loss(x, y)
        loss.backward()
        self.optimizer.step()

        return log

    def __init__(self, net, trainloader, optimizer, epoch_count, eval_every, save_path, evaluator, log_msg):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer

        self.numb_of_epochs = epoch_count
        self.eval_every = eval_every
        self.save_path = save_path
        self.evaluator = evaluator
        self.log_msg = log_msg

    def train(self):
        logger.info("Start training...")
        writer = SummaryWriter(self.save_path)

        for epoch in range(self.numb_of_epochs):  # loop over the dataset multiple times
            running_loss = {}
            for data in self.trainloader:
                # training step
                loss = self.training_step(data, epoch)

                # print statistics
                for key, value in loss.items():
                    running_loss[key] = (running_loss[key] + value) if key in running_loss else 0

            if epoch % self.eval_every == self.eval_every-1:  # print every K epochs
                logger.info('epoch: {}, tr_loss: {:4f}'.format(epoch, running_loss['loss'] / self.eval_every))

                for key, value in running_loss.items():
                    writer.add_scalar(DataModes.TRAINING + '/' + key, value, epoch)
                self.evaluator.evaluate(epoch, writer)
                running_loss = 0.0

        logger.info("... end of training!")


