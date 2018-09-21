import numpy as np
from gnuradio import gr


class TaggedSamplesPicker(gr.basic_block):

    def __init__(self, tag_key='here'):
        gr.basic_block.__init__(
            self,
            name=self.__class__.__name__,
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.tag_key = tag_key
        self.set_tag_propagation_policy(gr.TPP_CUSTOM)

    def general_work(self, input_items, output_items):
		
        n_output_asked = len(output_items[0])

        tags = self.get_tags_in_window(0, 0, len(input_items[0]))
        
        ks = list(map(lambda tag: tag.offset - self.nitems_read(0), tags))

        if len(ks) <= n_output_asked:
			# It can't produce as many output as asked:
            n_output_produced = len(ks)
            self.consume(0, len(input_items[0]))
        else:
			# It was given more inputs than needed:
			n_output_produced = n_output_asked
			self.consume(0, tags[ks[-1]].offset)

        output_items[0][:n_output_produced] = input_items[0][ks[:n_output_produced]]
        return n_output_produced
