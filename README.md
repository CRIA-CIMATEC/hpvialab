<h3 align="center">HP VIALAB</h3>

---

<p align="center"> Synthetic dataset for scanned images with multiple defects
		<br> 
</p>

## 📝 Contents table

- [About](#About)
- [Folder structure](#getting_started)
- [Autores](#authors)

## 🧐 About <a name = "About"></a>

This study proposes a new synthetic dataset for the removal of low-light, glare, wrinkled paper, color temperature variation and shadow and evaluates its performance with some state of the art image enhancement and denoising neural networks.

The results was achieved as a partnership between HP Inc. R&D Brazil and SENAI CIMATEC, using incentives of the Brazilian Informatics Law (Law n°. 8.2.48 of 1991).

## 🏁 Folder structure <a name = "getting_started"></a>

Folders / files and descriptions:
```
datasets/                                : Folder that contains the dataset with synthetic defects and its ground truth.
    synthetic_defects/                   : Folder that contains the synthetic and ground-truth dataset.
        {gt_position}.jpg                : Ground-truth image name template. `gt_position` means 'ground-truth image position' and ranges from 0 to 9899.
        {gt_position}_{syn_position}.jpg : Synthetic image name template. `gt_position` means 'ground-truth image position' and ranges from 0 to 9899. `syn_position` means 'synthetic image position' and ranges from 0 to 2.
```

## ✍️ Autores <a name = "authors"></a>

Developed by HP VIALAB at SENAI CIMATEC.

Contributors:
- Coordinator: Prof. Dr. Erick Giovani Sperandio Nascimento
- Coordinator: Ingrid Winkler
- Victor Rocha Santos
- Tiago Pagano
- Rafael Borges
- Lucas Kirsten
- Lucas Ortega
- Neilton Melgaço
- José Vinícius Dantas Paranhos
