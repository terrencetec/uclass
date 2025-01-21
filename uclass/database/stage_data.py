"""Stage data class"""
import numpy as np


class StageData:
    """Stage data"""
    def __init__(self, classifier, division, database="MongoDB"):
        """Constructor

        Parameters
        ----------
        classifier : str
            The classifier, e.g. "23-01".
        division : str
            The division, choose from
            ["opn", "lo", "co", "ltd", "pcc", "prod", "ss", "l10", "rev"].
        database : str
            Database to be used.
            Choose from ["MongoDB", "USPSA"].
            Default "MongoDB".
        """
        self.classifier = classifier
        self.division = division

        if database == "MongoDB":
            import uclass.database.mongo
            self.database = uclass.database.mongo.Mongo()
        else:
            raise ValueError(f"Database {database} not supported.")

    @property
    def classifier(self):
        """Classifier"""
        return self._classifier

    @classifier.setter
    def classifier(self, _classifier):
        """classifier.setter"""
        self._classifier = _classifier

    @property
    def division(self):
        """Division"""
        return self._division

    @division.setter
    def division(self, _division):
        """division.setter"""
        self._division = _division

    @property
    def hf(self):
        """Hit factors"""
        return self.get_hf()

    # @hf.setter
    # def hf(self, _hf):
    #     """hf.setter"""
    #     self._hf = _hf

    @property
    def database(self):
        """Database"""
        return self._database

    @database.setter
    def database(self, _database):
        """database.setter"""
        self._database = _database

    def get_hf(self, include_zeros=False):
        """Get hit factors
        
        Parameters
        ----------
        include_zeros : bool, optional
            Include zero hit factor.
            Default False.
        """
        hf = self.database.get_hf(self.classifier, self.division)
        hf = np.array(hf)
        if not include_zeros:
            hf = hf[hf>0]
        return hf

