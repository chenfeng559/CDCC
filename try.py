def train_one_epoch(
        self, 
        data_loader, 
        ano_probs, 
        replacing_data,
        inj,
        args
    ):
        '''
        ano_probs : list of probabilities of 12 types of inject anomalies
        replacing_data : data for soft_replacing method
        inj : InjectAnomalies instance
        args : include init parameters for inj and anomalies
        '''
        # 注意，由于混合多个数据集，因此没办法在dataloader之外写离散列的参数
        # 如果需要，可以在得到dataloader的时候记录来源数据集，并进一步在这里读取
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for x in tqdm(data_loader):
            x = x.type(torch.float32).cpu().numpy() # x : (B, S, F)
            anos = self._get_anomaly_types(ano_probs, x.shape[0])
            x_inj = []
            x_labels = []
            for x_ori, ano in zip(x, anos):
                if ano == 'no_anomaly':
                    x_inj.append(torch.tensor(x_ori).to(self.device))
                    x_labels.append(torch.tensor(np.zeros(x.shape[1])).to(self.device))
                    continue

                if ano == 'length_adjusting':
                    raise ValueError('Not Implemented Length-adjustment, due to the encapsulation of the data loader, unable to obtain data after the window.')
                
                self.logger.info(f'Anomaly type is {ano}')
                x_ano, label = inj.inject_anomalies(
                    T=x_ori,
                    cat_cols=np.array([]),
                    anomaly_type=ano,
                    rep_data=replacing_data,
                    **args.__dict__
                )
                
                x_inj.append(torch.tensor(x_ano).to(self.device))
                x_labels.append(torch.tensor(label).to(self.device))
            
            x = torch.stack(x_inj).type(torch.float32)
            x_labels = torch.stack(x_labels).type(torch.float32)

            rec = self.model(
                x, 
                torch.ones_like(x).type(torch.int32).to(x.device),
                torch.zeros(x.shape[:-1]).type(torch.int32).to(x.device)
            )
            loss = self.loss_func(rec, x)

            for k, v in zip(loss_dict, [loss]):
                loss_dict[k] += v.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip_norm)
            self.optimizer.step()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict

