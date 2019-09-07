def write_parameters(model_name, param_dic):
    file_name = model_name + "_params.log"
    f = open(file_name, "w+")

    for x in param_dic:
        f.write(x + "=" + str(param_dic[x]) + "\n")

    f.close()


def write_batch(model_name, epoch, batch, accuracy, loss):
    file_name = model_name + "_loss.log"
    f = open(file_name, "a+")
    f.write("epoch=" + str(epoch) + ", batch=" + str(batch) +
            ", accuracy=" + str(accuracy) + ", loss=" + str(loss) + "\n")
    f.close()