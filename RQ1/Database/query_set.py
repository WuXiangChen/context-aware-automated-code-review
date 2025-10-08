# 本节的主要目标是 集中保存所有用于查询的query

get_all_valid_ref_node_query = [
    {
        '$lookup': {
            'from': 'BaseProjectInfoGraphQL', 
            'let': {
                'repo': '$repo', 
                'ghid': '$ghid'
            }, 
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$eq': [
                                        '$owner/repo', '$$repo'
                                    ]
                                }, {
                                    '$eq': [
                                        '$number', '$$ghid'
                                    ]
                                }
                            ]
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0, 
                        'repo/owner': 0, 
                        'repo/owner:NUM': 0, 
                        'owner/repo': 0, 
                        'owner/repo:NUM': 0
                    }
                }
            ], 
            'as': 'result'
        }
    }, {
        '$set': {
            'number': {
                '$arrayElemAt': [
                    '$result.number', 0
                ]
            }, 
            'title': {
                '$arrayElemAt': [
                    '$result.title', 0
                ]
            }, 
            'state': {
                '$arrayElemAt': [
                    '$result.state', 0
                ]
            }, 
            'createdAt': {
                '$arrayElemAt': [
                    '$result.createdAt', 0
                ]
            }, 
            'updatedAt': {
                '$arrayElemAt': [
                    '$result.updatedAt', 0
                ]
            }, 
            'closedAt': {
                '$arrayElemAt': [
                    '$result.closedAt', 0
                ]
            }, 
            'mergedAt': {
                '$arrayElemAt': [
                    '$result.mergedAt', 0
                ]
            }, 
            'body': {
                '$arrayElemAt': [
                    '$result.body', 0
                ]
            }, 
            'author': {
                '$arrayElemAt': [
                    '$result.author', 0
                ]
            }, 
            'labels': {
                '$arrayElemAt': [
                    '$result.labels', 0
                ]
            }, 
            'type_ip': {
                '$arrayElemAt': [
                    '$result.type', 0
                ]
            }
        }
    }, {
        '$unset': 'result'
    }
]



get_all_ref_node_query = [
    {
        '$addFields': {
            'repo_id': {
                '$concat': [
                    {
                        '$ifNull': [
                            '$repo', ''
                        ]
                    }, ':', {
                        '$toString': {
                            '$ifNull': [
                                '$ghid', ''
                            ]
                        }
                    }
                ]
            }
        }
    }, {
        '$project': {
            '_id': 0
        }
    }
]