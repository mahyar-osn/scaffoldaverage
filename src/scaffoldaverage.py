import os

from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.zinc.status import OK as ZINC_OK

import numpy as np
from sklearn import decomposition


if __name__ == '__main__':
    config = {}
    config["root"] = "D:/sparc/codes/maplcinet_workflows/wholemount/non-cut"
    config["subjects"] = ['fitting-179', 'fitting-193', 'fitting-229', 'fitting-231', 'fitting-281', 'fitting-283',
                          'fitting-284', 'fitting-46', 'fitting-58']
    config["input_filename"] = "fit.exf"
    config["output_dir"] = "average"
    config["output_filename"] = "average-v2.exf"

    scaffold_node_list = []

    for subject in config["subjects"]:
        filename = os.path.join(config["root"], subject, config["input_filename"])
        context = Context("average")
        region = context.getDefaultRegion()
        result = region.readFile(filename)
        if result != ZINC_OK:
            print("read file is not OK!")

        field_module = region.getFieldmodule()
        field_module.beginChange()
        coordinates = field_module.findFieldByName('coordinates')
        # coordinates = discover_coordinate_field(field_module)
        source_fe_field = coordinates.castFiniteElement()
        cache = field_module.createFieldcache()

        nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_template = nodes.createNodetemplate()
        node_iter = nodes.createNodeiterator()
        node = node_iter.next()

        node_list = []

        while node.isValid():
            node_template.defineFieldFromNode(source_fe_field, node)
            cache.setNode(node)

            temp = []
            for derivative in [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2]:
                result, values = source_fe_field.getNodeParameters(cache, -1, derivative, 1, 3)
                temp.append(values)

            node_list.append(np.asarray(temp).T.tolist())

            node = node_iter.next()

        field_module.endChange()

        scaffold_node_list.append(np.asarray(node_list).reshape(117, 3).tolist())

    X = np.asarray(scaffold_node_list).reshape(9, 117*3)
    pca = decomposition.PCA(n_components=9)
    pca.fit(X)
    mean = pca.mean_
    components = pca.components_.T
    variance = pca.explained_variance_

    """
    
    To find the score (weight coefficient of shape i:
    
    shape_score_i = dot((X[i] - mean), components) # (X is your original data matrix)
    shape_score_i_normalized = (shape_score_i - mean) / s.d.
    
    To get a new shape:
    new_shape = mean + sum(PC_j * W_j) # (j is 1..N for N principal components)
    
    """

    mean = mean.reshape(117, 3)
    mean = mean.reshape(117 // 3, 3, 3)

    field_module = region.getFieldmodule()
    field_module.beginChange()
    source_fe_field = coordinates.castFiniteElement()
    cache = field_module.createFieldcache()
    nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    node_template = nodes.createNodetemplate()
    node_iter = nodes.createNodeiterator()
    node = node_iter.next()

    counter = 0
    while node.isValid():
        node_template.defineFieldFromNode(source_fe_field, node)
        cache.setNode(node)

        result = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, mean[counter].T[0].tolist())
        if result != ZINC_OK:
            print("X was not set")
        result = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, mean[counter].T[1].tolist())
        if result != ZINC_OK:
            print("ds1 was not set")
        result = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, mean[counter].T[2].tolist())
        if result != ZINC_OK:
            print("ds2 was not set")

        counter += 1
        node = node_iter.next()

    print(scaffold_node_list)
    region.writeFile(os.path.join(config["root"], config["output_filename"]))
    print("done")
