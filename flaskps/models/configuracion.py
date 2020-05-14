class Configuracion(object):

    db = None

    @classmethod
    def get_information(cls):
        sql = 'SELECT * FROM configuracion;'
        cursor = cls.db().cursor()
        cursor.execute(sql)
        return cursor.fetchone()
    
    @classmethod
    def get_users_rols(cls):
        sql = 'SELECT usuario_id, rol_id FROM usuario_tiene_rol'
        cursor = cls.db.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    @classmethod
    def deactive(cls):
        sql = 'UPDATE configuracion SET habilitado = 0 WHERE id = 1;'
        cursor = cls.db.cursor()
        cursor.execute(sql)
        cls.db.commit()
        return True

    @classmethod
    def active(cls):
        sql = 'UPDATE configuracion SET habilitado = 1 WHERE id = 1;'
        cursor = cls.db.cursor()
        cursor.execute(sql)
        cls.db.commit()
        return True

    @classmethod
    def get_current_status(cls):
        sql = 'SELECT habilitado FROM configuracion WHERE id = 1'
        cursor = cls.db.cursor()
        cursor.execute(sql)
        f = cursor.fetchone()['habilitado']
        return f

    @classmethod
    def get_page_size(cls):
        sql = 'SELECT tamanio_paginado FROM paginado'
        cursor = cls.db.cursor()
        cursor.execute(sql)
        pag = cursor.fetchone()
        return pag['tamanio_paginado']

    @classmethod
    def change_page_size(cls, size):
        sql = 'UPDATE paginado SET tamanio_paginado = %s WHERE id = 1'
        cursor = cls.db.cursor()
        cursor.execute(sql, (size))
        cls.db.commit()
        return  True

    @classmethod
    def edit_information(cls, data):
        sql = 'UPDATE configuracion SET titulo = %s, descripcion = %s, mail_orquesta = %s WHERE id = 1'
        data = (data.get('title'),data.get('description'),data.get('mail_orquesta'))
        cursor = cls.db.cursor()
        cursor.execute(sql, data)
        cls.db.commit()
        return  True