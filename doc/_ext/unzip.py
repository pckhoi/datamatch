import os
import zipfile

from sphinx.util.docutils import SphinxDirective


class Unzip(SphinxDirective):
    """Unzip a file to a folder relative to working directory

    Usage:
        .. unzip:: <zip_filename_relative_to_doc> <folder_relative_to_working_dir>
    """
    required_arguments = 2

    def run(self):
        file_path, extract_dir = self.arguments[0], self.arguments[1]
        _, filename = self.env.relfn2path(
            file_path, self.env.docname
        )
        if not os.path.isdir(extract_dir):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        return []


def setup(app):
    app.add_directive("unzip", Unzip)

    return {
        'version': '0.1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
