"""Mongo database class"""
import pymongo.mongo_client
import pandas
import numpy as np


class Mongo(pymongo.mongo_client.MongoClient):
    """Mongo database class"""
    def __init__(self, uri="mongodb://localhost:27017"):
        """Constructor

        Parameters
        ----------
        uri : str
            uri address of the mongodb instance
            Default "mongodb://localhost:27017"
        """
        super().__init__(uri)
        self.uri = uri

    @property
    def uri(self):
        """uri"""
        return self._uri

    @uri.setter
    def uri(self, _uri):
        """uri.setter"""
        self._uri = _uri
    
    def get_hf(self, classifier, division):
        """Get hit factors
        
        Parameters
        ----------
        classifier : str
            The classifier, e.g. "23-01".
        division : str
            The division, choose from
            ["opn", "lo", "co", "ltd", "pcc", "prod", "ss", "l10", "rev"].

        Returns
        -------
        hf : list
            Hit factors.
        """
        db = self["zeta"].scores  #FIXME hardcoded "zeta"
        query = [{'$match': {'classifier': classifier}},
                 {'$match': {'division': division}}]

        hf = []
        for item in list(db.aggregate(query)):
            if "bad" in item.keys():
                if item["bad"]:
                    continue
            hf.append(item["hf"])

        return hf
    
    def ping(self):
        """pings hitfactor.info mongo database to check connection"""
        try:
            self.admin.command('ping')
        except Exception as e:
            raise e
        else:
            return "Pinged deployment; successfully connected to MongoDB!"

                


